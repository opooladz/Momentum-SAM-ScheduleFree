import torch
import math
from typing import List, Tuple, Optional, Iterable, Dict, Any
from torch import Tensor

class AdamWMSAMScheduleFree(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        rho: float = 0.05,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho value: {rho}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, rho=rho,
            warmup_steps=warmup_steps, r=r, weight_lr_power=weight_lr_power,
            k=0, train_mode=True, weight_sum=0.0, lr_max=-1.0
        )
        super(AdamWMSAMScheduleFree, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('train_mode', True)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
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
            lr = lr * sched

            bias_correction1 = 1 - beta1 ** (k + 1)
            bias_correction2 = 1 - beta2 ** (k + 1)
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1

            lr_max = group['lr_max'] = max(lr, group['lr_max'])
            weight = ((k + 1) ** r) * (lr_max ** weight_lr_power)
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight
            ckp1 = weight / weight_sum if weight_sum != 0 else 0

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['z'] = p.clone().detach()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                z = state['z']

                # Update the steps for each param group update
                state['step'] += 1

                # AdamW-style update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Compute the MSAM ascent direction
                ascent_direction = z - p
                ascent_norm = ascent_direction.norm()
                normalized_ascent = ascent_direction / (ascent_norm + 1e-12)

                # Perform ascent step (SAM-like)
                p.add_(normalized_ascent, alpha=rho)

                # Perform descent step
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Weight decay
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

                # Update z (schedule-free momentum-like step)
                z.lerp_(p, 1 - ckp1).add_(grad, alpha=-lr * beta1)

            group['k'] = k + 1

        return loss

    @torch.no_grad()
    def move_to_ascent(self):
        for group in self.param_groups:
            rho = group['rho']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'z' in state:
                    ascent_direction = state['z'] - p
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
                    ascent_direction = state['z'] - p
                    ascent_norm = ascent_direction.norm()
                    normalized_ascent = ascent_direction / (ascent_norm + 1e-12)
                    p.sub_(normalized_ascent, alpha=rho)

    def train(self):
        for group in self.param_groups:
            group['train_mode'] = True
            beta1 = group['betas'][0]
            for p in group['params']:
                state = self.state[p]
                if 'z' in state:
                    p.data.lerp_(state['z'], 1 - beta1)

    def eval(self):
        for group in self.param_groups:
            group['train_mode'] = False
            beta1 = group['betas'][0]
            for p in group['params']:
                state = self.state[p]
                if 'z' in state:
                    p.data.lerp_(state['z'], 1 - 1/beta1)
