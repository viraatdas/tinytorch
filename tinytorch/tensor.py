import numpy as np
from numba import njit


@njit
def add_arrays(x, y):
    return x + y

@njit
def multiply_arrays(x, y):
    return x * y

@njit
def matmul_arrays(x, y):
    return x @ y



class Tensor:
    def __init__(self, data, requires_grad=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float16)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        # topological order of the graph nodes
        visited = set()
        topological_order = []

        def build_topo(v):
            if v is not visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topological_order.append(v)
        
        build_topo(self)
        
        for v in reversed(topological_order):
            v._backward()
    
    def __add__(self, other):
        


                
        