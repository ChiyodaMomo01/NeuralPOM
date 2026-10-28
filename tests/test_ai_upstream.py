"""EMA gradients and parameter identity, and dual-stage early-stop configuration."""
import unittest
from types import SimpleNamespace
import torch
from neuralpom.ai_baselines.models import SynchronizedEMAQuantizer
from neuralpom.ai_baselines.config import resolve_config

class OfficialAdapterTests(unittest.TestCase):
 def test_ema_buffers_stable_and_straight_through(self):
  torch.manual_seed(7)
  original=SimpleNamespace(_embedding_dim=3,_num_embeddings=5,_decay=.99,_epsilon=1e-5,
    _commitment_cost=.25,_embedding=torch.nn.Embedding(5,3),_ema_w=torch.nn.Parameter(torch.randn(5,3)),
    _ema_cluster_size=torch.zeros(5))
  model=SynchronizedEMAQuantizer(original)
  identifiers={n:id(v) for n,v in model.named_buffers()}
  x=torch.randn(2,3,4,4,requires_grad=True)
  old=model.embedding.clone();old_w=model.ema_w.clone()
  flat=x.permute(0,2,3,1).reshape(-1,3)
  indices=(flat.square().sum(1,keepdim=True)+old.square().sum(1)-2*flat@old.T).argmin(1)
  enc=torch.nn.functional.one_hot(indices,5).float()
  counts=.01*enc.sum(0);n=counts.sum();expected_counts=(counts+1e-5)/(n+5e-5)*n
  expected=(.99*old_w+.01*enc.T@flat.detach())/expected_counts[:,None]
  _,y,_,_=model(x)
  torch.testing.assert_close(model.embedding,expected)
  torch.testing.assert_close(y.permute(0,2,3,1).reshape(-1,3),enc@old)
  y.sum().backward();torch.testing.assert_close(x.grad,torch.ones_like(x))
  model(x.detach())
  self.assertEqual(identifiers,{n:id(v) for n,v in model.named_buffers()})
  self.assertFalse(list(model.parameters()))
  model.eval();before={n:v.clone() for n,v in model.named_buffers()};model(x.detach())
  for n,v in model.named_buffers():torch.testing.assert_close(v,before[n],rtol=0,atol=0)
 def test_both_stages_stop_and_all_architectures_registered(self):
  for name in ("triton","resunet","simvp","convlstm","pastnet"):
   config=resolve_config(dict(data_root="/data",run_root="/run",model=dict(architecture=name)))
   self.assertEqual(config["stages"]["pretrain"]["patience"],5)
   self.assertEqual(config["stages"]["posttrain"]["patience"],5)
  with self.assertRaises(ValueError):resolve_config(dict(data_root="/data",run_root="/run",stages=dict(posttrain=dict(patience=0))))

 def test_posttrain_plateau_saves_last_and_auxiliary_best(self):
  import tempfile, json
  from pathlib import Path
  from unittest.mock import patch
  from neuralpom.ai_baselines import engine
  with tempfile.TemporaryDirectory() as directory:
   config=resolve_config(dict(data_root="/data",run_root=directory,world_size=1,wandb=dict(mode="disabled")))
   runtime=SimpleNamespace(config=config,codec=SimpleNamespace(),contract=lambda:dict(identity="fixture"),
     encoded=lambda member,day:torch.ones(1,2,2),data=SimpleNamespace(load=lambda member,day,device:torch.ones(1,2,2)))
   def update(runtime,model,current,target,optimizer,clip):
    optimizer.zero_grad();prediction=model(current[None])[0];loss=(prediction-target).square().mean()
    loss.backward();optimizer.step()
    return prediction.detach(),loss.detach(),torch.tensor(1.)
   with patch.object(engine,"AIRuntime",return_value=runtime),patch.object(engine,"build_model",side_effect=lambda settings,codec:torch.nn.Conv2d(1,1,1)),patch.object(engine,"stage_samples",return_value=[(0,1000)]),patch.object(engine,"validation",return_value=1.),patch.object(engine,"local_daily_update",side_effect=update):
    engine.train(config,"pretrain",0,1,torch.device("cpu"))
    engine.train(config,"posttrain",0,1,torch.device("cpu"))
   root=Path(directory)
   for stage,step in (("pretrain",6),("posttrain",60)):
    record=json.loads((root/stage/"COMPLETE.json").read_text())
    self.assertEqual(record["epochs_completed"],6)
    self.assertEqual(record["step"],step)
    self.assertEqual(record["termination"],"validation_plateau")
   last=torch.load(root/"frozen/posttrain.pt",weights_only=False)
   best=torch.load(root/"frozen/posttrain_best.pt",weights_only=False)
   self.assertEqual(last["step"],60);self.assertEqual(best["step"],10)
