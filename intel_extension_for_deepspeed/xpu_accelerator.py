import torch
from deepspeed.accelerator.abstract_accelerator import DeepSpeedAccelerator
import intel_extension_for_pytorch as ipex  # noqa: F401
import oneccl_bindings_for_pytorch  #noqa: F401


class XPU_Accelerator(DeepSpeedAccelerator):
    def __init__(self):
        self._name = 'xpu'
        self._communication_backend_name = 'ccl'
        self.DoubleTensor = torch.xpu.DoubleTensor
        self.LongTensor = torch.xpu.LongTensor
        self.FloatTensor = torch.xpu.FloatTensor
        self.BFloat16Tensor = torch.xpu.BFloat16Tensor
        self.HalfTensor = torch.xpu.HalfTensor
        self.IntTensor = torch.xpu.IntTensor
        self.ByteTensor = torch.xpu.ByteTensor

    # Device APIs
    def device_name(self, device_index=None):
        if device_index == None:
            return 'xpu'
        return 'xpu:{}'.format(device_index)

    def device(self, device_index=None):
        return torch.xpu.device(device_index)

    def set_device(self, device_index):
        torch.xpu.set_device(device_index)

    def current_device(self):
        return torch.xpu.current_device()

    def current_device_name(self):
        return 'xpu:{}'.format(torch.xpu.current_device())

    def device_count(self):
        return torch.xpu.device_count()

    def synchronize(self, device_index=None):
        return torch.xpu.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.xpu.random

    def set_rng_state(self, new_state, device_index=None):
        return torch.xpu.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        if device_index == None:
            return torch.xpu.get_rng_state()
        return torch.xpu.get_rng_state(device_index)

    def manual_seed(self, seed):
        return torch.xpu.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.xpu.manual_seed_all(seed)

    def initial_seed(self, seed):
        return torch.xpu.initial_seed(seed)

    def default_generator(self, device_index):
        return torch.xpu.default_generators[device_index]

    # Streams/Events
    def Stream(self, device=None, priority=0, **kwargs):
        return torch.xpu.Stream(device, priority, **kwargs)

    def StreamContext(self, stream):
        return torch.xpu.StreamContext(stream)

    def stream(self, stream):
        return torch.xpu.stream(stream)

    def current_stream(self, device_index=None):
        return torch.xpu.current_stream(device_index)

    def default_stream(self, device_index=None):
        # torch.xpu does not support the sync behavior of default stream as cuda
        # use current_stream as workaround
        # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams
        return torch.xpu.current_stream(device_index)

    def Event(self, **kwargs):
        return torch.xpu.Event(**kwargs)

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
        return torch.xpu.is_available()

    def range_push(self, msg):
        return torch.xpu.itt.range_push(msg)

    def range_pop(self):
        return torch.xpu.itt.range_pop()

    def lazy_call(self, callback):
        return torch.xpu.lazy_init._lazy_call(callback)

    def communication_backend_name(self):
        return self._communication_backend_name

    # Data types
    def is_bf16_supported(self):
        return True

    def is_fp16_supported(self):
        return True

    # Tensor operations
    def pin_memory(self, tensor):
        return tensor.pin_memory(device=self.current_device_name())

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('xpu:'):
            return True
        else:
            return False

    def create_op_builder(self, op_name):
        from intel_extension_for_deepspeed.op_builder import CPUAdagradBuilder, CPUAdamBuilder, FusedAdamBuilder, QuantizerBuilder,  TransformerBuilder, UtilsBuilder
        from deepspeed.ops.op_builder import AsyncIOBuilder, SparseAttnBuilder
        if op_name == "AsyncIOBuilder":
            return AsyncIOBuilder()
        elif op_name == "CPUAdagradBuilder":
            return CPUAdagradBuilder()
        elif op_name == "CPUAdamBuilder":
            return CPUAdamBuilder()
        elif op_name == "FusedAdamBuilder":
            return FusedAdamBuilder()
        elif op_name == "QuantizerBuilder":
            return QuantizerBuilder()
        elif op_name == "SparseAttnBuilder":
            return SparseAttnBuilder()
        elif op_name == "TransformerBuilder":
            return TransformerBuilder()
        elif op_name == "UtilsBuilder":
            return UtilsBuilder()
        else:
            return None

    def build_extension(self):
        try:
            from intel_extension_for_pytorch.xpu.cpp_extension import DpcppBuildExtension
        except ImportError:
            from intel_extension_for_pytorch.xpu.utils import DpcppBuildExtension
        return DpcppBuildExtension
