"""Eager-only advancement that preserves independent closure subclasses."""
from neuralpom.hybrid.sparse_physics import SparsePhysics
from .smagorinsky import SeparatedSmagorinsky


class BaselinePhysics(SparsePhysics):
    def __init__(self,device,*,core_name):
        super().__init__(device,core_name=core_name,acceleration='eager',
                         compiled_kernels=(),checkpoint_physics=False,cg_backend='torch')

    def _configure(self,core):
        if not isinstance(core,SeparatedSmagorinsky):
            raise TypeError('BaselinePhysics requires a registered closure subclass')
        cls=type(core)
        # Parent installs exactly the registered production solver flags, but
        # its implicit branch changes __class__ to select the accelerated core.
        # Keep those flags while retaining our deliberate closure overrides.
        super()._configure(core)
        core.__class__=cls
