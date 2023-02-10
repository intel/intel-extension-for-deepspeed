import torch
from deepspeed.accelerator.abstract_accelerator import DeepSpeedAccelerator
import oneccl_bindings_for_pytorch  #noqa: F401


# accelerator for Intel CPU
class _CPU_Accelerator(DeepSpeedAccelerator):
    def __init__(self):
        self._name = 'cpu'
        self._communication_backend_name = 'ccl'

    # Device APIs
    def device_name(self, device_index=None):
        return 'cpu'

    def device(self, device_index=None):
        return torch.xpu.device(device_index)

    def set_device(self, device_index):
        return

    def current_device(self):
        return 0

    def current_device_name(self):
        return 'cpu'

    def device_count(self):
        return 1

    def synchronize(self, device_index=None):
        return

    # RNG APIs
    def random(self):
        return torch.xpu.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index == None :
            return torch.xpu.set_rng_state(new_state)
        return torch.xpu.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        return torch.get_rng_state()

    def manual_seed(self, seed):
        return torch.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.xpu.manual_seed_all(seed)

    def initial_seed(self, seed):
        return torch.xpu.initial_seed(seed)

    def default_generator(self, device_index):
        return torch.default_generator

    # Streams/Events
    @property
    def Stream(self):
        return

    def stream(self, stream):
        return torch.xpu.stream(stream)

    def current_stream(self, device_index=None):
        return torch.xpu.current_stream(device_index)

    def default_stream(self, device_index=None):
        # torch.xpu does not support the sync behavior of default stream as cuda
        # use current_stream as workaround
        # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams
        return torch.xpu.current_stream(device_index)

    @property
    def Event(self):
        return None

    # Memory management
    def empty_cache(self):
        return torch.xpu.empty_cache()

    def memory_allocated(self, device_index=None):
        return torch.xpu.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        return torch.xpu.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        return torch.xpu.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        return torch.xpu.memory_reserved(device_index)

    def max_memory_cached(self, device_index=None):
        return torch.xpu.max_memory_reserved(device_index)

    def reset_max_memory_cached(self, device_index=None):
        return torch.xpu.reset_max_memory_reserved(device_index)

    def memory_stats(self, device_index=None):
        return torch.xpu.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        return torch.xpu.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        return torch.xpu.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        return torch.xpu.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        return torch.xpu.get_device_properties(device_index).total_memory

    # Misc
    def amp(self):
        return torch.xpu.amp

    def is_available(self):
        return True

    def range_push(self, msg):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_push(msg)
        return

    def range_pop(self):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_pop()
        return

    def lazy_call(self, callback):
        return callback()

    def communication_backend_name(self):
        return self._communication_backend_name

    # Data types
    def is_bf16_supported(self):
        return True

    def is_fp16_supported(self):
        return True

    # Tensor operations

    @property
    def BFloat16Tensor(self):
        return torch.BFloat16Tensor

    @property
    def ByteTensor(self):
        return torch.ByteTensor

    @property
    def DoubleTensor(self):
        return torch.DoubleTensor

    @property
    def FloatTensor(self):
        return torch.FloatTensor

    @property
    def HalfTensor(self):
        return torch.HalfTensor

    @property
    def IntTensor(self):
        return torch.IntTensor

    @property
    def LongTensor(self):
        return torch.LongTensor

    def pin_memory(self, tensor):
        return tensor.pin_memory(device=self.current_device_name())

    def op_builder_dir(self):
        return "intel_extension_for_deepspeed.op_builder"

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('xpu:'):
            return True
        else:
            return False

    # create an instance of op builder and return, name specified by class_name 
    def create_op_builder(self, op_name):
        builder_class = self.get_op_builder(op_name)
        if builder_class != None:
            return builder_class()
        return None

    # return an op builder class, name specified by class_name
    def get_op_builder(self, class_name):
        from intel_extension_for_deepspeed.op_builder import CPUAdagradBuilder, CPUAdamBuilder, FusedAdamBuilder, QuantizerBuilder, TransformerBuilder, UtilsBuilder
        from deepspeed.ops.op_builder.async_io import AsyncIOBuilder
        from deepspeed.ops.op_builder.sparse_attn import SparseAttnBuilder

        if class_name == "AsyncIOBuilder":
            return AsyncIOBuilder
        elif class_name == "CPUAdagradBuilder":
            return CPUAdagradBuilder
        elif class_name == "CPUAdamBuilder":
            return CPUAdamBuilder
        elif class_name == "FusedAdamBuilder":
            return FusedAdamBuilder
        elif class_name == "QuantizerBuilder":
            return QuantizerBuilder
        elif class_name == "SparseAttnBuilder":
            return SparseAttnBuilder
        elif class_name == "TransformerBuilder":
            return TransformerBuilder
        elif class_name == "UtilsBuilder":
            return UtilsBuilder
        else:
            return None

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension
