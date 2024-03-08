from hmac import new
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn

__factory_optim = {
    'adam': optim.Adam,
    'sgd': optim.SGD,
}

__args_dict_optim = {
    'adam': {'lr': 0.0005, 'betas': (0.9, 0.98)},
    'sgd': {'lr': 0.001},
}

__factory_scheduler = {
    'linear': lr_scheduler.LinearLR,
    'exponential': lr_scheduler.ExponentialLR,
    "reduce": lr_scheduler.ReduceLROnPlateau,
    'none': None,
}

__args_dict_scheduler = {
    'linear': {'start_factor': 1.0, 'end_factor': 0.1, 'total_iters': 200},
    'exponential': {'gamma': 0.93}, # 0.99
    'reduce': {"factor": 0.1, "patience": 4, "threshold": 0.01, "cooldown": 0, "min_lr": 0.000001, "mode": "min"},
    'none': {},
}


def create_optimizer(name: str, params: nn.Parameter, **kwargs):
    assert(name in __factory_optim), 'invalid optimizer_name'
    _kwargs = {k: v for k, v in kwargs.items() if k in __args_dict_optim[name]}
    default_kwargs = __args_dict_optim[name]
    new_kwargs = {**default_kwargs, **_kwargs}
    optimizer = __factory_optim[name](params, **new_kwargs)

    return optimizer, new_kwargs


def create_scheduler(name: str, optimizer: optim.Optimizer, **kwargs):
    assert(name in __factory_scheduler), 'invalid scheduler_name'
    _kwargs = {k: v for k, v in kwargs.items() if k in __args_dict_scheduler[name]}
    default_kwargs = __args_dict_scheduler[name]
    new_kwargs = {**default_kwargs, **_kwargs}
    scheduler = __factory_scheduler[name](optimizer, **new_kwargs) if name != 'none' else None

    return scheduler, new_kwargs


if __name__ == "__main__":
    optimizer, _ = create_optimizer('adam', nn.Linear(10, 10).parameters())
    optimizer, _ = create_optimizer('sgd', nn.Linear(10, 10).parameters())
    lr_scheduler, _ = create_scheduler('linear', optimizer)
    lr_scheduler, _ = create_scheduler('exponential', optimizer)
    lr_scheduler, _ = create_scheduler('none', optimizer)
    print(optimizer)
    print(lr_scheduler)