"""Backend-specific utilities.

This subpackage provides backend-specific utilities for the ``zea`` library. Most backend logic is handled by Keras 3, but a few features require custom wrappers to ensure compatibility and performance across JAX, TensorFlow, and PyTorch.

.. note::
    Most backend-specific logic is handled by Keras 3, so this subpackage is intentionally minimal. Only features not natively supported by Keras (such as JIT and autograd) are implemented here.

Key Features
------------

- **JIT Compilation** (:func:`zea.backend.jit`):
  Provides a unified interface for just-in-time (JIT) compilation of functions, dispatching to the appropriate backend (JAX or TensorFlow) as needed. This enables accelerated execution of computationally intensive routines. Note that jit compilation is not yet supported when using the `torch` backend.

- **Automatic Differentiation** (:class:`zea.backend.AutoGrad`):
  Offers a backend-agnostic wrapper for automatic differentiation, allowing gradient computation regardless of the underlying ML library.

- **Backend Submodules:**

  - :mod:`zea.backend.jax` -- JAX-specific utilities and device management.
  - :mod:`zea.backend.torch` -- PyTorch-specific utilities and device management.
  - :mod:`zea.backend.tensorflow` -- TensorFlow-specific utilities, and device management, as well as data loading utilities.

- **Data Loading** (:func:`zea.backend.tensorflow.make_dataloader`):
  This function is implemented using TensorFlow's efficient data pipeline utilities. It provides a convenient way to load and preprocess data for machine learning workflows, leveraging TensorFlow's ``tf.data.Dataset`` API.

"""

"""Backend-specific utilities."""

from contextlib import nullcontext
import keras
from zea import log

# --- 导入辅助函数 ---
def _import_tf():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        return None

def _import_jax():
    try:
        import jax
        return jax
    except ImportError:
        return None

def _import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None

tf_mod = _import_tf()
jax_mod = _import_jax()
torch_mod = _import_torch()

# --- JIT 编译分发 ---
def jit(func=None, jax=True, tensorflow=True, **kwargs):
    if func is None:
        def decorator(func):
            return _jit_compile(func, jax=jax, tensorflow=tensorflow, **kwargs)
        return decorator
    else:
        return _jit_compile(func, jax=jax, tensorflow=tensorflow, **kwargs)

def _jit_compile(func, jax=True, tensorflow=True, **kwargs):
    backend = keras.backend.backend()

    if backend == "tensorflow" and tensorflow:
        if tf_mod is None: raise ImportError("TensorFlow not installed.")
        return tf_mod.function(func, jit_compile=kwargs.pop("jit_compile", True), **kwargs)
    elif backend == "jax" and jax:
        if jax_mod is None: raise ImportError("JAX not installed.")
        return jax_mod.jit(func, **kwargs)
    elif backend == "torch":
        # 新增：PyTorch Compile 支持
        if torch_mod and hasattr(torch_mod, "compile"):
            # 获取用户可能传入的 fullgraph 参数，默认为 False 以提高兼容性
            fullgraph = kwargs.get("fullgraph", False)
            try:
                # 使用 inductor 后端是目前的通用最佳实践
                return torch_mod.compile(func, backend="inductor", fullgraph=fullgraph)
            except Exception as e:
                log.warning(f"torch.compile failed for {func.__name__}: {e}. Running in eager mode.")
                return func
        return func
    else:
        return func

# --- 设备管理上下文 ---
class on_device:
    def __init__(self, device: str):
        self.device = self.get_device(device)
        self.context = self.get_context(self.device)

    def get_device(self, device: str):
        if device is None: return None
        backend = keras.backend.backend()
        if backend == "torch":
            return device.replace("gpu", "cuda")
        return device

    def get_context(self, device):
        if device is None: return nullcontext()
        backend = keras.backend.backend()
        if backend == "torch":
            import torch
            return torch.device(device)
        # ... (保留原有的 tf/jax 逻辑如果还需要，否则可简化) ...
        return nullcontext()

    def __enter__(self):
        # PyTorch 的 device 上下文通常不如 TF 那么全局，
        # 但这里为了兼容性可以保留，或者让用户手动 to(device)
        if self.context:
            self.context.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.context:
            self.context.__exit__(exc_type, exc_val, exc_tb)

# --- 关键：导出 Data Loader ---
if keras.backend.backend() == "tensorflow":
    from zea.backend.tensorflow.dataloader import make_dataloader
elif keras.backend.backend() == "torch":
    # 这里导入我们新建的 PyTorch 版本
    from zea.backend.torch.dataloader import make_dataloader
else:
    # 默认 fallback，防止 import 报错
    def make_dataloader(*args, **kwargs):
        raise NotImplementedError(f"No dataloader for backend {keras.backend.backend()}")
