# BFF_Optimizer: a pure Bfloat16 optimizer - basic idea is we use Kahan summarization to offset the Bfloat16 precision reduction, allowing full training in BFloat16.

# paper credit - "Revisiting Bfloat16 training" - https://arxiv.org/abs/2010.06192
# original inspiration - https://github.com/arogozhnikov/adamw_bfloat16
# Kahan summation - https://en.wikipedia.org/wiki/Kahan_summation_algorithm

import torch
from torch.optim.optimizer import Optimizer


class BFF_Optimizer(Optimizer):
    def __init__(
        self,
        params,
        pure_mode=False,
        optimizer_states16=True,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
    ):

        """
        Args:
                params (iterable): iterable of parameters to optimize or dicts defining
                    parameter groups
                lr (float, optional): learning rate (default: 1e-3)
                betas (Tuple[float, float], optional): coefficients used for computing
                    running averages of gradient and its square (default: (0.9, 0.999))
                eps (float, optional): term added to the denominator to improve
                    numerical stability (default: 1e-8)
                weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        """
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            pure_mode=pure_mode,
            optimizer_states16=optimizer_states16,
        )

        super().__init__(params, defaults)
        print(f"BFF Optimizer initialized with {defaults}")

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            pure_mode = group["pure_mode"]
            pure_state = group["optimizer_states16"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("BFF does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    if pure_mode:
                        assert (
                            p.dtype == torch.bfloat16
                        ), "BFF requires BFloat16 datatype"

                    state["step"] = torch.tensor(0.0)

                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.bfloat16)

                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.bfloat16
                    )

                    # Kahan summation - accumulated error tracker
                    if pure_mode:
                        state["compensation"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                # main processing

                # update the steps for each param group update
                state["step"] += 1

                grad = p.grad
                # exp_avg = state["exp_avg"]

                # exp_avg_sq = state["exp_avg_sq"]

                # Decay the first and second moment running average coefficient
                """exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                """

                state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(
                    grad, grad.conj(), value=1 - beta2
                )

                if pure_mode:
                    compensation = state["compensation"]

                # weight decay, AdamW style - todo - this differs from torch impl
                if weight_decay:
                    p.data.mul_(1 - lr * weight_decay)

                denom_correction = (1 - beta2 ** state["step"]) ** 0.5

                # lr update to compensation
                if pure_mode:
                    compensation.addcdiv_(
                        state["exp_avg"],
                        state["exp_avg_sq"].sqrt().add_(group["eps"], alpha=1),
                        value=-lr * denom_correction,
                    )

                    # update weights with compensation (Kahan summation)
                    # save error back to compensation for next iteration
                    if pure_mode:
                        buffer = p.clone()
                        p.add_(compensation)
                        compensation.add_(buffer.sub_(p))

                else:
                    # if group['weight_decay'] != 0:
                    #    grad.add_(group['weight_decay'], p.data)
                    # step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                    # p.data.addcdiv_(-step_size, exp_avg, denom)
                    p.data.addcdiv_(
                        state["exp_avg"],
                        state["exp_avg_sq"].sqrt().add_(group["eps"], alpha=1),
                        value=-lr * denom_correction,
                    )
