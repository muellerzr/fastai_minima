
# fastai_minima
> A mimal version of fastai with the barebones needed to work with Pytorch


```python
#all_slow
```

## Install

`pip install fastai_minima`

## How to use

This library is designed to bring in only the _minimal_ needed from [fastai](https://github.com/fastai/fastai) to work with raw Pytorch. This includes:

* Learner
* Callbacks
* Optimizer
* DataLoaders (but not the `DataBlock`)
* Metrics

Below we can find a very minimal example based off my [Pytorch to fastai, Bridging the Gap](https://muellerzr.github.io/fastblog/2021/02/14/Pytorchtofastai.html) article:

```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

dset_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

dset_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(dset_train, batch_size=4,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(dset_test, batch_size=4,
                                         shuffle=False, num_workers=2)
```

    Files already downloaded and verified
    Files already downloaded and verified


```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

```python
criterion = nn.CrossEntropyLoss()
```

```python
from torch import optim
from fastai_minima.optimizer import OptimWrapper
from fastai_minima.learner import Learner, DataLoaders
from fastai_minima.callback.training import CudaCallback, ProgressCallback
```

```python
def opt_func(params, **kwargs): return OptimWrapper(optim.SGD(params, **kwargs))

dls = DataLoaders(trainloader, testloader)
```

```python
learn = Learner(dls, Net(), loss_func=criterion, opt_func=opt_func)

# To use the GPU, do 
# learn = Learner(dls, Net(), loss_func=criterion, opt_func=opt_func, cbs=[CudaCallback()])
```

```python
learn.fit(2, lr=0.001)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.269467</td>
      <td>2.266472</td>
      <td>01:20</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.876898</td>
      <td>1.879593</td>
      <td>01:21</td>
    </tr>
  </tbody>
</table>


    /mnt/d/lib/python3.7/site-packages/torch/autograd/__init__.py:132: UserWarning: CUDA initialization: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx (Triggered internally at  /pytorch/c10/cuda/CUDAFunctions.cpp:100.)
      allow_unreachable=True)  # allow_unreachable flag


If you want to do differential learning rates, when creating your `splitter` to pass into fastai's `Learner` you should utilize the `convert_params` to make it compatable with Pytorch Optimizers:

```python
def splitter(m): return convert_params([[m.a], [m.b]])
```
```python
learn = Learner(..., splitter=splitter)
```
