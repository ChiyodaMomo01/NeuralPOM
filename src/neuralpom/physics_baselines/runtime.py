"""Pure physics runtime; no neural weights, normalization, or training cache."""
import torch
from neuralpom.v1.runtime import Runtime
from neuralpom.v1.core_spec import core_spec
from neuralpom.v1.data import FixedCoreDataset
from neuralpom.v1.checkpoint import capture
from .smagorinsky import build_smagorinsky_core,closure_contract
from .physics import BaselinePhysics


class SmagorinskyRuntime(Runtime):
    def __init__(self,config,device,*,cm=.05,cc=.01):
        self.config,self.device=config,torch.device(device)
        self.spec=core_spec(config['core'])
        self.interval_seconds=config['interval_seconds']
        self.physical_forward_steps=0
        grid=config['grid']
        self.core=build_smagorinsky_core(self.spec.name,self.device,grid['im'],grid['jm'],cm=cm,cc=cc)
        self.physics=BaselinePhysics(self.device,core_name=self.spec.name)
        self.physics._configure(self.core)
        self.baseline=capture(self.core,cpu=False)
        self.data=FixedCoreDataset(config['data_root'],self.interval_seconds,self.spec.name)
        if self.data.grid != (grid['im'],grid['jm'],self.core.kb):
            raise ValueError('Dataset and baseline grid differ')
        self.normalizer=None

    def contract(self):
        return dict(version='neuralpom-pure-physics-baseline-v1',
                    closure=closure_contract(self.core),
                    reference_data_identity=self.data.data_identity,
                    physical_step_seconds=self.spec.physical_step_seconds,
                    external_step_seconds=self.spec.external_step_seconds,
                    interval_seconds=self.interval_seconds,grid=self.config['grid'],
                    alpha=self.core.alpha,smoth=self.core.smoth,
                    split=self.config['split'],acceleration='eager',neural_weights=None)

    def correct(self,*args,**kwargs):
        raise RuntimeError('Pure physics baselines cannot apply neural corrections')
