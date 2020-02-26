# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR, PolyCosineAnnealingLR, WarmupPolynormialLR


class OptimizerDict(dict):

    def __init__(self, *args, **kwargs):
        super(OptimizerDict, self).__init__(*args, **kwargs)

    def state_dict(self):
        return [optim.state_dict() for optim in self.values()]

    def load_state_dict(self, state_dicts):
        for state_dict, optim in zip(state_dicts, self.values()):
            optim.load_state_dict(state_dict)
            # for state in optim.state.values():
            #     for k, v in state.items():
            #         if isinstance(v, torch.Tensor):
            #             state[k] = v.cuda()


def make_optimizer(cfg, model):
    params = []
    if cfg.DARTS_ON:
        a_params = []
        for key, value in model.named_parameters():
            if 'arch' in key:
                a_params.append(value)
                continue
            params.append(value)
        optim_w = torch.optim.SGD(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            momentum=cfg.SOLVER.MOMENTUM)
        optim_a = torch.optim.Adam(
            a_params,
            lr=cfg.DARTS.LR_A,
            weight_decay=cfg.DARTS.WD_A)
        return OptimizerDict(optim_w=optim_w,
                             optim_a=optim_a)
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if key.startswith("rpn.head.rec"):
            if not key.endswith("scale"):
                lr *= cfg.SOLVER.ONE_STAGE_HEAD_LR_FACTOR
            else:
                print("do not apply SOLVER.ONE_STAGE_HEAD_LR_FACTOR to {}".format(key))
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.SCHEDULER == 'multistep':
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif cfg.SOLVER.SCHEDULER == 'poly':
        return WarmupPolynormialLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            cfg.SOLVER.POLY_POWER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )


def make_search_lr_scheduler(cfg, optimizer_dict):
    optimizer = optimizer_dict['optim_w']

    return PolyCosineAnnealingLR(
        optimizer,
        max_iter=cfg.SOLVER.MAX_ITER,
        T_max=cfg.DARTS.T_MAX,
        eta_min=cfg.DARTS.LR_END
    )
