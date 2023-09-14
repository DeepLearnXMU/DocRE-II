from torch.nn import Module
import operator
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.scatter_gather import gather
from torch.nn.parallel._functions import Scatter
from torch._utils import (
    _get_all_device_indices,
    _get_available_device_type,
    _get_device_index,
    _get_devices_properties
)
import torch
import pdb
from itertools import chain

def is_namedtuple(obj):
    # Check if type was created from collections.namedtuple or a typing.NamedTuple.
    return (isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields"))

def _check_balance(device_ids):
    imbalance_warn = """
    There is an imbalance between your GPUs. You may want to exclude GPU {} which
    has less than 75% of the memory or cores of GPU {}. You can do so by setting
    the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
    environment variable."""
    device_ids = [_get_device_index(x, True) for x in device_ids]
    dev_props = _get_devices_properties(device_ids)

    def warn_imbalance(get_prop):
        values = [get_prop(props) for props in dev_props]
        min_pos, min_val = min(enumerate(values), key=operator.itemgetter(1))
        max_pos, max_val = max(enumerate(values), key=operator.itemgetter(1))
        if min_val / max_val < 0.75:
            warnings.warn(imbalance_warn.format(device_ids[min_pos], device_ids[max_pos]))
            return True
        return False

    if warn_imbalance(lambda props: props.total_memory):
        return
    if warn_imbalance(lambda props: props.multi_processor_count):
        return


def scatter1(inputs, target_gpus, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    pdb.set_trace()
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if is_namedtuple(obj):
            return [type(obj)(*args) for args in zip(*map(scatter_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(scatter_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None
    return res

def scatter(inputs, target_gpus, dim=0):
    Len=inputs['input_ids'].size(0)
    #
    # GPU0 = int(Len * 0.4)
    # # GPU1 = int(Len * 0.4)
    # # GPU1=int(Len*0.45)
    #
    # # GPU0=Len-GPU1*2
    # # Len_list=[GPU0,GPU1+GPU0,GPU1*2+GPU0,Len]
    # Len_list = [GPU0, Len]
    # # print("===============================")
    # # print(Len_list)
    # # print("===============================")
    if(Len>10):
        Len_list = [int(Len/2), Len]
    else:
        Len_list = [2, 5]


    res=[]
    start=0
    for i in range(len(target_gpus)):
        with torch.cuda.device(target_gpus[i]):
            temp={}
            for key in inputs.keys():
                temp[key]=inputs[key][start:Len_list[i]]
                if isinstance(temp[key], torch.Tensor):
                    temp[key]=temp[key].cuda()

        start=Len_list[i]
        res.append(temp)
    return res


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class DataParallel(Module):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__()

        device_type = _get_available_device_type()
        if device_type is None:
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = _get_all_device_indices()

        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = [_get_device_index(x, True) for x in device_ids]
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device(device_type, self.device_ids[0])

        _check_balance(self.device_ids)

        if len(self.device_ids) == 1:
            self.module.to(self.src_device_obj)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        # for forward function without any inputs, empty list and dict will be created
        # so the module can be executed on one device which is the first one in device_ids
        if not inputs and not kwargs:
            inputs = ((),)
            kwargs = ({},)

        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids, not torch.is_grad_enabled())

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)