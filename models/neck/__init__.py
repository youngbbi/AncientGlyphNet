from .FPN import FPN
from .FPEM_FFM import FPEM_FFM

__all__ = ['build_neck']
support_neck = ['FPN', 'FPEM_FFM']


def build_neck(neck_name, **kwargs):
    assert neck_name in support_neck, f'all support neck is {support_neck}'
    neck = eval(neck_name)(**kwargs)
    return neck
