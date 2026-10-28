from __future__ import annotations

import torch


DEFAULT_CORE_DTE = 20.0
DEFAULT_CORE_ISPLIT = 30
DEFAULT_PROBLEM_ID = 4


def instantiate_core(core_cls, im: int, jm: int, problem_id: int = DEFAULT_PROBLEM_ID):
    try:
        core = core_cls(im=im, jm=jm, problem_id=problem_id)
    except TypeError:
        core = core_cls(im=im, jm=jm)
        core.iproblem = problem_id
    return core


def configure_core_runtime(core, dte: float = DEFAULT_CORE_DTE, isplit: int = DEFAULT_CORE_ISPLIT) -> None:
    core.dte = dte
    core.isplit = isplit
    core.dti = core.dte * core.isplit
    core.dte2 = core.dte * 2.0
    core.dti2 = core.dti * 2.0
    core.ispi = 1.0 / core.isplit
    core.isp2i = 1.0 / (2.0 * core.isplit)


def build_initialized_core(
    core_cls,
    device: torch.device,
    im: int,
    jm: int,
    problem_id: int = DEFAULT_PROBLEM_ID,
    dte: float = DEFAULT_CORE_DTE,
    isplit: int = DEFAULT_CORE_ISPLIT,
):
    core = instantiate_core(core_cls, im=im, jm=jm, problem_id=problem_id)
    core.device = torch.device(device)
    # Constructors may allocate on the default CUDA device. Honor CPU and
    # non-default DDP devices before case setup consumes any existing buffers.
    for name, value in list(vars(core).items()):
        if torch.is_tensor(value) and value.device != core.device:
            setattr(core, name, value.to(core.device))
    configure_core_runtime(core, dte=dte, isplit=isplit)
    core.run_initialization()
    return core
