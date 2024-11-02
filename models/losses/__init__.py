import copy
from .DB_loss import DBLoss

__all__ = ['build_loss']
support_loss = ['DBLoss']

def build_loss(config):
    copy_config = copy.deepcopy(config)
    loss_type = copy_config.pop('type')
    assert loss_type in support_loss, f'all support loss is {support_loss}'
    criterion = eval(loss_type)(**copy_config)
    return criterion
