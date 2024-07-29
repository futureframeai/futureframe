import logging

from torch import optim

log = logging.getLogger(__name__)


def get_constant_lr_scheduler(optimizer, lr: float, *args, **kwargs):
    """
    Returns a learning rate scheduler that keeps the learning rate constant.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        lr (float): The base learning rate.

    Returns:
        torch.optim.lr_scheduler.ConstantLR: The learning rate scheduler.
    """
    return optim.lr_scheduler.ConstantLR(optimizer, lr)


def get_linear_warmup_cos_lr_scheduler(
    optimizer, max_steps: int, lr: float, start_factor=0.3, end_factor=0.1, warmup_fraction=0.02, *args, **kwargs
):
    """
    Returns a learning rate scheduler that combines linear warmup and cosine annealing.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        max_steps (int): The total number of training steps.
        lr (float): The base learning rate.
        start_factor (float, optional): The factor by which to multiply the base learning rate at the start of the warmup phase.
        end_factor (float, optional): The factor by which to multiply the base learning rate at the end of the cosine annealing phase.
        warmup_fraction (float, optional): The fraction of total steps to use for linear warmup.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: The learning rate scheduler.
    """
    total_warmup_iters = int(warmup_fraction * max_steps)
    total_cosine_iters = int(max_steps * (1 - warmup_fraction))

    scheduler1 = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        total_iters=total_warmup_iters,
    )

    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_cosine_iters,
        eta_min=lr * end_factor,
    )

    lr_scheduler = optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
    return lr_scheduler


lr_scheduler_registry = dict(
    constant=get_constant_lr_scheduler,
    linear_warmup_cos=get_linear_warmup_cos_lr_scheduler,
)


def create_lr_scheduler(lr_scheduler_name: str, **kwargs):
    """
    Create a learning rate scheduler based on the given name.

    Args:
        lr_scheduler_name (str): The name of the learning rate scheduler.

    Returns:
        The learning rate scheduler corresponding to the given name.

    Raises:
        KeyError: If the learning rate scheduler with the given name is not found in the registry.
    """
    try:
        return lr_scheduler_registry[lr_scheduler_name](**kwargs)
    except KeyError as e:
        log.error(f"Learning rate scheduler {lr_scheduler_name} not found in the registry.")
        raise e
