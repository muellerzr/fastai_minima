# Contains code used/modified by fastai_minima author from fastai
# Copyright 2019 the fast.ai team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language

from fastcore.basics import patch, range_of
from fastcore.dispatch import retain_type
from fastcore.foundation import L
from fastcore.meta import use_kwargs_dict
from fastcore.xtras import is_listy
import sys
import torch
from torch import nn
from torch import Tensor, as_tensor

import os

import pandas as pd
import numpy as np
from numpy import ndarray, array

import torch
from torch.nn.parallel import DistributedDataParallel

from types import SimpleNamespace
from typing import Iterable, Generator

defaults = SimpleNamespace()
defaults.__doc__= "fastai defaults"

NoneType = type(None)

defaults.use_cuda = None

def params(m):
    "Return all parameters of `m`"
    return [p for p in m.parameters()]

def default_device(use_cuda=-1):
    "Return or set default device; `use_cuda`: None - CUDA if available; True - error if not available; False - CPU"
    if use_cuda != -1: defaults.use_cuda=use_cuda
    use = defaults.use_cuda or (torch.cuda.is_available() and defaults.use_cuda is None)
    assert torch.cuda.is_available() or not use
    return torch.device(torch.cuda.current_device()) if use else torch.device('cpu')

def to_device(b, device=None):
    "Recursively put `b` on `device`."
    if defaults.use_cuda==False: device='cpu'
    elif device is None: device=default_device()
    def _inner(o): return o.to(device, non_blocking=True) if isinstance(o,Tensor) else o.to_device(device) if hasattr(o, "to_device") else o
    return apply(_inner, b)


def to_concat(xs, dim=0):
    "Concat the element in `xs` (recursively if they are tuples/lists of tensors)"
    if not xs: return xs
    if is_listy(xs[0]): return type(xs[0])([to_concat([x[i] for x in xs], dim=dim) for i in range_of(xs[0])])
    if isinstance(xs[0],dict):  return {k: to_concat([x[k] for x in xs], dim=dim) for k in xs[0].keys()}
    #We may receive xs that are not concatenable (inputs of a text classifier for instance),
    #   in this case we return a big list
    try:    return retain_type(torch.cat(xs, dim=dim), xs[0])
    except: return sum([L(retain_type(o_.index_select(dim, tensor(i)).squeeze(dim), xs[0])
                          for i in range_of(o_)) for o_ in xs], L())

def maybe_gather(x, axis=0):
    "Gather copies of `x` on `axis` (if training is distributed)"
    if num_distrib()<=1: return x
    ndim = x.ndim
    res = [x.new_zeros(*x.shape if ndim > 0 else (1,)) for _ in range(num_distrib())]
    torch.distributed.all_gather(res, x.contiguous() if ndim > 0 else x[None])
    return torch.cat(res, dim=axis) if ndim > 0 else torch.cat(res, dim=axis).mean()

def apply(func, x, *args, **kwargs):
    "Apply `func` recursively to `x`, passing on args"
    if is_listy(x): return type(x)([apply(func, o, *args, **kwargs) for o in x])
    if isinstance(x,dict):  return {k: apply(func, v, *args, **kwargs) for k,v in x.items()}
    res = func(x, *args, **kwargs)
    return res if x is None else retain_type(res, x)

def to_detach(b, cpu=True, gather=True):
    "Recursively detach lists of tensors in `b `; put them on the CPU if `cpu=True`."
    def _inner(x, cpu=True, gather=True):
        if not isinstance(x,Tensor): return x
        x = x.detach()
        if gather: x = maybe_gather(x)
        return x.cpu() if cpu else x
    return apply(_inner, b, cpu=cpu, gather=gather)

def find_bs(b):
    "Recursively search the batch size of `b`."
    return item_find(b).shape[0]

def to_half(b):
    "Recursively map lists of tensors in `b ` to FP16."
    return apply(lambda x: x.half() if torch.is_floating_point(x) else x, b)

def distrib_barrier():
    "Place a synchronization barrier in distributed training"
    if num_distrib() > 1 and torch.distributed.is_initialized(): torch.distributed.barrier()
        
def num_distrib():
    "Return the number of processes in distributed training (if applicable)."
    return int(os.environ.get('WORLD_SIZE', 0))
        
def rank_distrib():
    "Return the distributed rank of this process (if applicable)."
    return int(os.environ.get('RANK', 0))

def item_find(x, idx=0):
    "Recursively takes the `idx`-th element of `x`"
    if is_listy(x): return item_find(x[idx])
    if isinstance(x,dict):
        key = list(x.keys())[idx] if isinstance(idx, int) else idx
        return item_find(x[key])
    return x

# Cell
def find_device(b):
    "Recursively search the device of `b`."
    return item_find(b).device

def is_iter(o):
    "Test whether `o` can be used in a `for` loop"
    #Rank 0 tensors in PyTorch are not really iterable
    return isinstance(o, (Iterable,Generator)) and getattr(o,'ndim',1)

def noop (x=None, *args, **kwargs):
    "Do nothing"
    return x

def get_model(model):
    "Return the model maybe wrapped inside `model`."
    return model.module if isinstance(model, (DistributedDataParallel, nn.DataParallel)) else model

def trainable_params(m):
    "Return all trainable parameters of `m`"
    return [p for p in m.parameters() if p.requires_grad]

norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LayerNorm)

# Cell
def norm_bias_params(m, with_bias=True):
    "Return all bias and BatchNorm parameters"
    if isinstance(m, norm_types): return L(m.parameters())
    res = L(m.children()).map(norm_bias_params, with_bias=with_bias).concat()
    if with_bias and getattr(m, 'bias', None) is not None: res.append(m.bias)
    return res

def distrib_barrier():
    "Place a synchronization barrier in distributed training"
    if num_distrib() > 1 and torch.distributed.is_initialized(): torch.distributed.barrier()

@patch
def __array_eq__(self:Tensor,b):
    return torch.equal(self,b) if self.dim() else self==b

def _array2tensor(x):
    if x.dtype==np.uint16: x = x.astype(np.float32)
    # windows default numpy int dytpe is int32, while torch tensor default int dtype is int64
    # https://github.com/numpy/numpy/issues/9464
    if sys.platform == "win32":
        if x.dtype==np.int: x = x.astype(np.int64)
    return torch.from_numpy(x)

@use_kwargs_dict(dtype=None, device=None, requires_grad=False, pin_memory=False)
def tensor(x, *rest, **kwargs):
    "Like `torch.as_tensor`, but handle lists too, and can pass multiple vector elements directly."
    if len(rest): x = (x,)+rest
    # There was a Pytorch bug in dataloader using num_workers>0. Haven't confirmed if fixed
    # if isinstance(x, (tuple,list)) and len(x)==0: return tensor(0)
    res = (x if isinstance(x, Tensor)
           else torch.tensor(x, **kwargs) if isinstance(x, (tuple,list))
           else _array2tensor(x) if isinstance(x, ndarray)
           else as_tensor(x.values, **kwargs) if isinstance(x, (pd.Series, pd.DataFrame))
           else as_tensor(x, **kwargs) if hasattr(x, '__array__') or is_iter(x)
           else _array2tensor(array(x), **kwargs))
    if res.dtype is torch.float64: return res.float()
    return res