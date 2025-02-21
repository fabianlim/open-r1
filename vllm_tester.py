import torch.distributed as dist
from datetime import timedelta
import os
from tqdm import trange
from time import sleep
import torch 
import numpy as np
from typing import List

# To be used if VLLM is generating 
class VLLMDeviceManager:

    def __init__(
        self, 
        process_index: int,
        local_process_index: int,
        vllm_device: str,
    ):

        # detect if we want sharding
        # new format auto:<SHARD>:<TP>
        self.mini_shards = 1
        self.tensor_parallel = 1
        self.process_index = process_index
        self.local_process_index = local_process_index
        self.device = f'cuda:{process_index}'

        # get the local world size
        # https://pytorch.org/docs/stable/elastic/run.html#environment-variables
        self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        assert self.local_world_size % self.mini_shards == 0, "number of mini shards must divide local world size"

        # NOTE: some draft code
        if vllm_device.startswith("auto:"):
            _,  self.mini_shards, self.tensor_parallel = vllm_device.split(":")
            self.mini_shards = int(self.mini_shards)
            self.tensor_parallel = int(self.tensor_parallel)

        self.mini_shard_size = self.local_world_size // self.mini_shards

        # NOTE: disable these checks first. until finalize how to handle the distributed devices
        # vllm_device = self.args.vllm_device
        # if vllm_device == "auto":
        #     if torch.cuda.device_count() == 1:
        #         vllm_device = "cuda:0"  # particular case when training with onyl 1 GPU: share it
        #     else:
        #         vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
        # # Check that the requested device is available
        # if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
        #     raise ValueError(
        #         f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
        #         "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
        #         "value lower than the number of GPUs available on your machine—typically, reducing it by one "
        #         f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
        #     )
        # # Check that the requested device is not also used for training
        # if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
        #     warnings.warn(
        #         f"The requested device {vllm_device} is also being used for training. For higher throughput "
        #         "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
        #         "If this is intentional, you may ignore this warning but should adjust "
        #         "`vllm_gpu_memory_utilization` accordingly."
        #     )
        
        # create a distributed group for communication on its mini-shard

        world_size = torch.distributed.get_world_size()
        self._group, subgroups  = torch.distributed.new_subgroups_by_enumeration(
            [
                list(range(i*self.mini_shard_size, (i+1) * self.mini_shard_size)) 
                for i in range(world_size // self.mini_shard_size)
            ]
        )

    def gather(self, tensor):
        # this follows accelerate.utils.operations.gather, which is an all 
        # gather operation, used to implement the gather op here.
        output_tensors = torch.empty(
            self.mini_shard_size * tensor.numel(),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        self._group_barrier()
        torch.distributed.all_gather_into_tensor(output_tensors, tensor, group=self._group)
        return output_tensors.view(-1, *tensor.size()[1:])

    @property
    def is_vllm_process(self):
        # essentially this is the mini shard leader
        return self.is_shard_leader

    @property
    def vllm_device(self):
        # NOTE: cannot handle TP for now
        # this indexes into the mini-shard local to the node
        local_mini_shard_index = self.local_process_index // self.mini_shard_size
        # FIXME: this one is to be changed when we consider TP and local shards
        return self.local_world_size + local_mini_shard_index

    @property
    def local_rank_mini_shard(self):
        # rank of this process within its mini shard
        return self.process_index % self.mini_shard_size

    @property
    def is_shard_leader(self):
        return self.local_rank_mini_shard == 0

    @property
    def global_rank_shard_leader(self):
        # the global rank of the shard leader
        mini_shard_idx = self.process_index // self.mini_shard_size
        return mini_shard_idx * self.mini_shard_size

    def _group_barrier(self):
        torch.distributed.barrier(
            group=self._group, device_ids=[self.process_index]
        )

    def gather_tensor_list(self, tensors: List[torch.tensor]):

        batch = len(tensors)
        assert batch > 0, "cannot gather empty tensor list"
        dtype = tensors[0].dtype

        # assume they are all the same dtype
        # dtype = tensors[0].dtype

        # assume the tensors are 1-D
        sizes = torch.tensor(
            [tensors[i].shape[-1] for i in range(len(tensors))],
            dtype=torch.int32, device=self.device
        )

        # get a single tensor
        self._group_barrier()
        gathered_sizes = self.gather(sizes)

        # for all_gather
        output_objects = [
            torch.empty(
                gathered_sizes[i*batch:(i+1)*batch].sum(),
                dtype=dtype, device=self.device
            )
            for i in range(self.mini_shard_size)
        ]

        # have to use gather because we cannot gaurantee
        # all tensors are of equal length
        self._group_barrier()
        torch.distributed.all_gather(
            output_objects,
            torch.cat(tensors), # form batch into single tensor
            group=self._group, 
        )

        # pretend like its a gather
        if not self.is_shard_leader:
            return None

        outputs = []
        for i in range(self.mini_shard_size):
            batch_sizes = gathered_sizes[i*batch:(i+1)*batch].tolist()
            outputs.extend(torch.split(
                output_objects[i], batch_sizes,
            ))
        return outputs

    def scatter_tensor_list(
        self, 
        batch: int,
        dtype,
        tensors: List[torch.tensor] = None,
    ):

        # assume all the ranks give the same batch size

        if tensors is not None:
            assert len(tensors) > 0, "cannot scatter empty tensor list"
            assert self.is_shard_leader, "only the shard leader can scatte"

            # assume the tensors are 1-D
            sizes = torch.tensor(
                [tensors[i].shape[-1] for i in range(len(tensors))],
                dtype=torch.int32, device=self.device
            )
        else:

            # need to broadcast the sizes
            sizes = torch.empty(
                batch * self.mini_shard_size, 
                dtype=torch.int32, device=self.device
            )

        # get all the sizes
        self._group_barrier()
        torch.distributed.broadcast(
            sizes,
            group=self._group, 
            src=self.global_rank_shard_leader,
        )

        # total number of tokens in one mini shard
        scattered_sizes_list = sizes.view(-1, batch).sum(axis=-1)

        # largest number of tokens in one rank of the mini shard
        max_size = scattered_sizes_list.max().item()

        # scatter tensor
        scattered_tensor = torch.empty(
            # scattered_sizes.sum(),
            max_size,
            dtype=dtype, device=self.device
        )

        # each rank will get a cat of its batch
        scattered_tensor_list = None
        if self.is_shard_leader:
            scattered_tensor_list = []
            for i in range(self.mini_shard_size):

                # collect all the tokens in the batch
                # of a rank in the minishard
                ids = []
                for j in range(batch):
                    ids.extend(tensors[i*batch+j])

                # pad it to make all same rank
                ids.extend([0] * (max_size - len(ids)))
                scattered_tensor_list.append(
                    torch.tensor(ids, dtype=dtype, device=self.device)
                )

        # get max_size tokens of the rank
        self._group_barrier()
        torch.distributed.scatter(
            scattered_tensor, scattered_tensor_list,
            group=self._group, 
            src=self.global_rank_shard_leader,
        )

        # get the number of tokens in this rank
        r = self.local_rank_mini_shard
        szs_sum = scattered_sizes_list[r]
        szs = sizes[r*batch:(r+1)*batch].tolist()

        # drop the padding and split
        outputs = torch.split(
            scattered_tensor[:szs_sum], szs
        )
        return outputs

if __name__ == '__main__':

    # Enviromnent variables
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=10))
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)

    manager = VLLMDeviceManager(
        process_index=rank,
        local_process_index=local_rank,
        vllm_device='auto:2:1'
    )
    mini_shard_size = manager.mini_shard_size

    batch_size = 4
    rng = np.random.default_rng(seed=1)
    NUM_TOKENS = rng.choice(100, size=(world_size, batch_size))
    TOKEN_IDS = []
    for r in range(world_size):
        TOKEN_IDS.append([
            rng.choice(1000, size=(sz,)) for sz in NUM_TOKENS[r]
        ])

    MINI_SHARD_SUMS = [
        sum([ # rank
            sum([ # seq
                sum(seq) for seq in TOKEN_IDS[i * mini_shard_size + j]
            ])
            for j in range(mini_shard_size)
        ])
        for i in range(world_size // mini_shard_size)
    ]

    for _ in trange(100):

        input_ids = [
            torch.tensor(ids, dtype=torch.int32, device=device)
            for ids in TOKEN_IDS[rank]
        ]

        # only the min_shard leader will get stuff
        gathered_ids = manager.gather_tensor_list(input_ids)

        if manager.is_vllm_process:
            mini_shard_index = rank // mini_shard_size
            assert (
                MINI_SHARD_SUMS[mini_shard_index]
                ==
                sum([ids.sum() for ids in gathered_ids])
            )

        # scatter back
        scattered_ids = manager.scatter_tensor_list(
            len(input_ids), input_ids[0].dtype,
            gathered_ids, 
        )

        assert (
            sum([len(x) for x in TOKEN_IDS[rank]]) 
            ==
            sum([len(x) for x in scattered_ids])
        )

        sleep(1)

    dist.destroy_process_group()


