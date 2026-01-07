"""Autograd wrapper for different backends."""
import functools
import keras
from . import _import_jax, _import_tf, _import_torch

tf = _import_tf()
jax = _import_jax()
torch = _import_torch()

class AutoGrad:
    def __init__(self, verbose=False):
        self.function = None
    
    @property
    def backend(self):
        return keras.backend.backend()

    def set_function(self, function):
        self.function = function

    def gradient(self, variable, **kwargs):
        variable = keras.ops.convert_to_tensor(variable)
        if self.backend == "torch":
            # 训练核心逻辑：确保梯度追踪
            if not variable.requires_grad:
                variable = variable.detach().requires_grad_(True)
            
            # 前向传播
            out = self.function(variable, **kwargs)
            
            # 反向传播计算梯度
            # create_graph=True 允许计算二阶导数（某些高级训练可能需要）
            gradients = torch.autograd.grad(out, variable, create_graph=True)[0]
            return gradients
        
        # ... (保留原有的 TF/JAX 逻辑，为了完整性建议保留)
        pass 

    def gradient_and_value(self, variable, has_aux: bool = False, **kwargs):
        variable = keras.ops.convert_to_tensor(variable)
        if self.backend == "torch":
            if not variable.requires_grad:
                variable = variable.detach().requires_grad_(True)
            
            if has_aux:
                out, aux = self.function(variable, **kwargs)
            else:
                out = self.function(variable, **kwargs)
            
            # 计算梯度
            gradients = torch.autograd.grad(out, variable, create_graph=True)[0]
            
            if has_aux:
                return gradients, (out, aux)
            return gradients, out
            
        raise NotImplementedError(f"Backend {self.backend} not fully implemented here.")

    def get_gradient_jit_fn(self):
        """Returns a compiled gradient function."""
        if self.backend == "torch":
            # 修改：不要抛出错误，直接返回原函数
            # PyTorch 的 autograd 本身就是高度优化的 C++ 实现
            return self.gradient
        
        # ... (其他后端逻辑)

    def get_gradient_and_value_jit_fn(self, has_aux: bool = False, disable_jit=False):
        func = lambda x, **kwargs: self.gradient_and_value(x, has_aux=has_aux, **kwargs)
        if disable_jit:
            return func
            
        if self.backend == "torch":
            # 修改：直接返回，不做 torch.compile 以避免动态图兼容性问题
            # 这里的稳定性优于微小的性能提升
            return func
            
        # ... (其他后端逻辑)
