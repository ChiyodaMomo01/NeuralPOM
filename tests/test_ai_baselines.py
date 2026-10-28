"""Semantic tests for complete state, exact sampling, and one-day gradients."""
import copy
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
import torch
from neuralpom.ai_baselines.config import resolve_config, plan
from neuralpom.ai_baselines.state import StateCodec, StateNormalizer, FIELDS_3D, FIELDS_2D
from neuralpom.ai_baselines.models import build_model
from neuralpom.ai_baselines.rollout import local_daily_update, forecast
from neuralpom.v1.config import resolve_config as hybrid_config
from neuralpom.v1.sampling import stage_samples
from neuralpom.evaluation.protocol import variants


class AIBaselineTests(unittest.TestCase):
    def test_complete_state_roundtrip(self):
        codec = StateCodec((8, 8, 16))
        self.assertEqual(codec.channels, 313)
        state = {name: torch.randn(8,8,16) for name in FIELDS_3D}
        state.update({name: torch.randn(8,8) for name in FIELDS_2D})
        encoded = codec.pack(state)
        norm = StateNormalizer(torch.randn(313), torch.rand(313)+.1)
        decoded = codec.unpack(norm.decode(norm.encode(encoded)))
        for name in state:
            torch.testing.assert_close(decoded[name], state[name], atol=2e-6, rtol=2e-6)
        del state['ub']
        with self.assertRaises(ValueError): codec.pack(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'stats.npz'; norm.save(path)
            torch.testing.assert_close(StateNormalizer.load(path).scale, norm.scale)

    def test_exact_hybrid_sampling(self):
        ai = resolve_config(dict(data_root='/data',run_root='/runs'))
        hybrid = hybrid_config(dict(data_root='/data',run_root='/hybrid'))
        for stage in ('pretrain','posttrain'):
            for split in ('train','validation'):
                self.assertEqual(stage_samples(ai,stage,split), stage_samples(hybrid,stage,split))
        self.assertEqual(plan(ai)['pretrain']['samples'], 7000)
        self.assertEqual(plan(ai)['posttrain']['samples'], 700)
        with self.assertRaises(ValueError): resolve_config(dict(data_root='/data',run_root='/runs', stages={'midtrain':{}}))
        with self.assertRaises(ValueError): resolve_config(dict(data_root='/data',run_root='/runs', stages={'posttrain':{'tbptt_hours':48}}))

    def test_one_day_gradient_and_actual_prediction_carry(self):
        # Independent analytic derivative: d(theta*x-target)^2/dtheta.
        model = torch.nn.Conv2d(1,1,1,bias=False)
        model.weight.data.fill_(.5)
        runtime = SimpleNamespace(predict=lambda network,x:network(x[None])[0],
            normalizer=SimpleNamespace(encode=lambda x:x),codec=SimpleNamespace(pack=lambda x:x,
            weights=lambda core,reference:torch.ones_like(reference)),core=None)
        initial = torch.full((1,2,2),2.,requires_grad=True)
        optimizer = torch.optim.SGD(model.parameters(),lr=.1)
        first,_,_ = local_daily_update(runtime,model,initial,torch.zeros_like(initial),optimizer,100)
        self.assertFalse(first.requires_grad); self.assertIsNone(initial.grad)
        torch.testing.assert_close(first,torch.ones_like(first))
        self.assertAlmostEqual(model.weight.item(), .1, places=5)
        second,_,_ = local_daily_update(runtime,model,first,torch.zeros_like(first),optimizer,100)
        self.assertAlmostEqual(model.weight.item(), .08, places=5)
        torch.testing.assert_close(second,torch.full_like(second,.1))
        self.assertFalse(second.requires_grad)
        outputs=list(forecast(runtime,model,initial,3))
        self.assertEqual(len(outputs),3)
        self.assertTrue(all(not value.requires_grad for _,value in outputs))

    def test_both_architectures_real_grid(self):
        codec=StateCodec()
        for cfg in ({'architecture':'triton','spatial_hidden_dim':8,'temporal_hidden_dim':16,
             'num_spatial_layers':2,'num_temporal_layers':2}, {'architecture':'resunet','hidden_dim':8}):
            network=build_model(cfg,codec).eval()
            with torch.no_grad(): output=network(torch.zeros(1,313,90,82))
            self.assertEqual(tuple(output.shape),(1,313,90,82))
            self.assertTrue(torch.isfinite(output).all())

    def test_paired_analysis_validates_ai_reference(self):
        import json
        from neuralpom.evaluation.analysis import collect, summarize
        from neuralpom.evaluation.plots import curve_appearance
        groups=[dict(id=f'{core}_{frequency}',core=core,frequency=frequency,config='/hybrid')
            for core in ('implicit_p0_fixed','explicit') for frequency in ('1h','1day')]
        with tempfile.TemporaryDirectory() as directory:
            protocol=dict(groups=groups,correction_policies=['raw','constrained','density75'],
                ai_baselines=[dict(id='ai',core='implicit_p0_fixed',config='/ai')],
                horizons_days=[1],cases=[dict(member=7,start_day=8000)],output_root=directory)
            for variant in variants(protocol):
                root=Path(directory)/'runs'/variant['id'];root.mkdir(parents=True)
                common=dict(data_identity={'root':'/reference'},split={'train_members':[0]},grid={'im':90,'jm':82})
                contract=dict(common,core=variant['core'])
                if variant.get('model_family')=='ai':
                    contract=dict(data_identity=common['data_identity'],split=common['split'],
                        reference_core=variant['core'],codec={'grid':[90,82,16]})
                (root/'manifest.json').write_text(json.dumps(dict(variant=variant,
                    training_contract=contract,weight_sha256=variant['stage'])))
                case=root/'m7_day8000';case.mkdir()
                (case/'status.json').write_text(json.dumps(dict(status='complete',completed_days=1)))
                (case/'daily.jsonl').write_text(json.dumps(dict(lead_day=1,time=8001,reference_time=8001,
                    metrics={f'mse/{f}':1. for f in ('u','v','t','s','el')}))+'\n')
            data=collect(protocol);result=summarize(protocol,data)
            self.assertEqual(len(result['ai_pairs']),2)
            self.assertEqual(curve_appearance(data['ai_posttrain_ai_based']['variant'])[2],':')
            path=Path(directory)/'runs/ai_posttrain_ai_based/manifest.json'
            value=json.loads(path.read_text());value['training_contract']['data_identity']={'root':'/wrong'}
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError): collect(protocol)

    def test_optional_campaign_registration(self):
        groups=[dict(id=f'{core}_{frequency}',core=core,frequency=frequency,config='/hybrid')
             for core in ('implicit_p0_fixed','explicit') for frequency in ('1h','1day')]
        old=dict(groups=groups,correction_policies=['raw','constrained','density75'])
        original=list(variants(old))
        self.assertEqual(len(original),40)
        new=dict(old,ai_baselines=[dict(id='ai_implicit',core='implicit_p0_fixed',config='/ai')])
        registered=list(variants(new));self.assertEqual(registered[:40],original)
        self.assertEqual([v['stage'] for v in registered[40:]],['pretrain','posttrain'])
        self.assertTrue(all(v['group_id']=='implicit_p0_fixed_1day' for v in registered[40:]))


if __name__=='__main__': unittest.main()
