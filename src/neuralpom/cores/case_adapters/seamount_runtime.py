"""Reference-state setup for the measured native open seamount protocol."""
from .fortran_compat import install as install_closure
from .seamount_boundary import install as install_boundary, VERSION


def install(core, family):
    core.alpha = .225 if family == 'explicit' else .333
    core.smoth = .1 if family == 'explicit' else .15
    # The registered experiment starts at rest with zero incident flow and
    # fixed initial exterior tracer profiles. Initialise from that same state.
    core.rfe = core.rfw = core.rfn = core.rfs = 1.
    core.uabe, core.uabw = core.uab[-2, :].clone(), core.uab[1, :].clone()
    core.vabn, core.vabs = core.va[:, -2].clone(), core.va[:, 1].clone()
    core.ele, core.elw = core.el[-1, :].clone(), core.el[0, :].clone()
    core.eln, core.els = core.el[:, -1].clone(), core.el[:, 0].clone()
    for key, name, axis, index in (
        ('tbe', 't', 0, -1), ('tbw', 't', 0, 0),
        ('sbe', 's', 0, -1), ('sbw', 's', 0, 0),
        ('tbn', 't', 1, -1), ('tbs', 't', 1, 0),
        ('sbn', 's', 1, -1), ('sbs', 's', 1, 0)):
        field = getattr(core, name)
        setattr(core, key, (field[index, :, :] if axis == 0 else field[:, index, :]).clone())
    install_closure(core)
    install_boundary(core, family)
    return VERSION
