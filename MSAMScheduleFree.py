import torch
from typing import Iterable, Union, Dict, Any

class MSAMScheduleFree(torch.optim.Optimizer):
    def __init__(
            self,
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
            lr: float = 1e-1,
            momentum: float = 0.9,
            weight_decay: float = 1e-2,
            rho: float = 0.3,
            warmup_steps: int = 0,
            r: float = 0.0,
            weight_lr_power: float = 2.0,
            ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum <= 0 or momentum >= 1:
            raise ValueError(f"Momentum must be between 0 and 1 exclusive: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if rho < 0.0:
            raise ValueError(f"Invalid rho value: {rho}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            rho=rho,
            k=0,
            warmup_steps=warmup_steps,
            r=r,
            weight_lr_power=weight_lr_power,
            train_mode=True,
            weight_sum=0.0,
            lr_max=-1.0,
        )
        super(MSAMScheduleFree, self).__init__(params, defaults)

    def eval(self):
        for group in self.param_groups:
            group['train_mode'] = False

    def train(self):
        for group in self.param_groups:
            group['train_mode'] = True

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if not group['train_mode']:
                raise Exception("Not in train mode!")

            momentum = group['momentum']
            weight_decay = group['weight_decay']
            rho = group['rho']
            k = group['k']
            warmup_steps = group['warmup_steps']
            r = group['r']
            weight_lr_power = group['weight_lr_power']

            # Schedule-free warmup and LR adjustment
            if k < warmup_steps:
                sched = (k + 1) / warmup_steps
            else:
                sched = 1.0
            lr = group['lr'] * sched

            lr_max = group['lr_max'] = max(lr, group['lr_max'])
            weight = ((k + 1) ** r) * (lr_max ** weight_lr_power)
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            try:
                ckp1 = weight / weight_sum
            except ZeroDivisionError:
                ckp1 = 0

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize z if not present
                if 'z' not in state:
                    state['z'] = torch.clone(p.data).detach()

                # Compute ascent direction (z - p)
                ascent_direction = state['z'] - p.data

                # Normalize ascent direction
                ascent_norm = ascent_direction.norm()
                normalized_ascent = ascent_direction / (ascent_norm + 1e-12)

                # Perform ascent step (SAM-like)
                p.add_(normalized_ascent, alpha=rho)

                # Apply weight decay
                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                # Update parameter (descent step)
                p.add_(grad, alpha=-lr)

                # Update z (momentum-like step from Schedule-Free SGD)
                state['z'].lerp_(p.data, 1 - ckp1)
                state['z'].sub_(grad, alpha=lr * momentum)

            group['k'] = k + 1

    @torch.no_grad()
    def move_to_ascent(self):
        for group in self.param_groups:
            rho = group['rho']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'z' in state:
                    ascent_direction = state['z'] - p.data
                    ascent_norm = ascent_direction.norm()
                    normalized_ascent = ascent_direction / (ascent_norm + 1e-12)
                    p.add_(normalized_ascent, alpha=rho)

    @torch.no_grad()
    def move_from_ascent(self):
        for group in self.param_groups:
            rho = group['rho']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'z' in state:
                    ascent_direction = state['z'] - p.data
                    ascent_norm = ascent_direction.norm()
                    normalized_ascent = ascent_direction / (ascent_norm + 1e-12)
                    p.sub_(normalized_ascent, alpha=rho)
