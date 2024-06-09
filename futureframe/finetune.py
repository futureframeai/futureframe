import pandas as pd
import torch


class SupervisedTrainCollator:
    def __init__(
        self,
        **kwargs,
    ):
        pass

    def __call__(self, data):
        if data[0][0] is not None:
            x_cat_input_ids = torch.cat([row[0] for row in data], 0)
        else:
            x_cat_input_ids = None

        if data[0][1] is not None:
            x_cat_att_mask = torch.cat([row[1] for row in data], 0)
        else:
            x_cat_att_mask = None

        if data[0][2] is not None:
            x_num = torch.cat([row[2] for row in data], 0)
        else:
            x_num = None

        col_cat_input_ids = data[0][3]
        col_cat_att_mask = data[0][4]
        num_col_input_ids = data[0][5]
        num_att_mask = data[0][6]
        y = None
        if data[0][7] is not None:
            y = pd.concat([row[7] for row in data])

        inputs = {
            "x_cat_input_ids": x_cat_input_ids,
            "x_cat_att_mask": x_cat_att_mask,
            "x_num": x_num,
            "col_cat_input_ids": col_cat_input_ids,
            "col_cat_att_mask": col_cat_att_mask,
            "num_col_input_ids": num_col_input_ids,
            "num_att_mask": num_att_mask,
        }
        return inputs, y


class LinearWarmupScheduler:
    def __init__(self, optimizer, base_lr, warmup_epochs, warmup_start_lr=-1, warmup_ratio=0.1, **kwargs):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs

        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else base_lr * warmup_ratio

    def step(self, cur_epoch):
        if cur_epoch < self.warmup_epochs:
            self._warmup_lr_schedule(
                step=cur_epoch,
                optimizer=self.optimizer,
                max_step=self.warmup_epochs,
                init_lr=self.warmup_start_lr,
                max_lr=self.base_lr,
            )
        elif cur_epoch == self.warmup_epochs:
            self._set_lr(self.optimizer, self.base_lr)

    def init_optimizer(self):
        self._set_lr(self.optimizer, self.warmup_start_lr)

    def _warmup_lr_schedule(self, optimizer, step, max_step, init_lr, max_lr):
        lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
        self._set_lr(optimizer, lr)

    def _set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
