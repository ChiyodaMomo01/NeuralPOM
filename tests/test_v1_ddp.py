"""Two real Gloo ranks: repeated checkpointed forwards and complete restart."""
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from neuralpom.v1.checkpoint import RestartStore
from neuralpom.v1.engine import rank_zero_action


class CheckpointedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)
        self.linear.weight.data.fill_(.3)

    def forward(self, value):
        return checkpoint(self.linear, value, use_reentrant=False)


def _rank(rank, init_file, directory):
    dist.init_process_group('gloo', rank=rank, world_size=2, init_method='file://' + init_file)
    try:
        model = DDP(CheckpointedLinear())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        x = torch.tensor([[float(rank + 1)]])
        losses = []
        for _ in range(3):
            x = x + model(x)
            losses.append(x.square().mean())
            x = checkpoint(lambda state: state * 2, x, use_reentrant=False)
        loss = torch.stack(losses).mean()
        loss.backward()
        expected_weight = torch.tensor(.3, requires_grad=True)
        expected = []
        for other in range(2):
            value = torch.tensor(float(other + 1))
            terms = []
            for _ in range(3):
                value = value + expected_weight * value
                terms.append(value.square())
                value = value * 2
            expected.append(torch.stack(terms).mean())
        torch.stack(expected).mean().backward()
        torch.testing.assert_close(model.module.linear.weight.grad.flatten()[0], expected_weight.grad)
        optimizer.step()
        store = RestartStore(directory, {'stage': 'midtrain3'})
        store.save(model, optimizer, {'epoch': 1, 'batch': 2}, 9, physical={'rank_value': rank})
        restored = DDP(CheckpointedLinear())
        restored_optimizer = torch.optim.AdamW(restored.parameters())
        value, physical = store.load(restored, restored_optimizer)
        torch.testing.assert_close(restored.module.linear.weight, model.module.linear.weight, rtol=0, atol=0)
        assert value['step'] == 9 and physical['rank_value'] == rank
        try:
            rank_zero_action(lambda: (_ for _ in ()).throw(ValueError('tracking failure')))
        except RuntimeError as exc:
            assert 'tracking failure' in str(exc)
        else:
            raise AssertionError('Every rank must receive the tracking failure')
        import neuralpom.v1.checkpoint as storage
        original_atomic = storage.atomic_torch
        def fail_rank_write(path, value):
            if rank == 1 and Path(path).name == 'rank_1.pt':
                raise OSError('simulated rank shard write failure')
            return original_atomic(path, value)
        storage.atomic_torch = fail_rank_write
        try:
            store.save(model, optimizer, {'epoch': 1, 'batch': 3}, 10)
        except RuntimeError as exc:
            assert 'rank shard write failure' in str(exc)
        else:
            raise AssertionError('All ranks must observe an incomplete checkpoint generation')
        finally:
            storage.atomic_torch = original_atomic
        value, _ = store.load(model, optimizer)
        assert value['step'] == 9
        import json
        from types import SimpleNamespace
        from neuralpom.v1.engine import Training
        probe = Training.__new__(Training)
        probe.runtime = SimpleNamespace(device=torch.device('cpu'))
        probe.world, probe.rank, probe.step = 2, rank, 1
        probe.cursor = {'epoch': 0, 'batch': 0, 'window': 1}
        probe.output = Path(directory)
        probe.logger = SimpleNamespace(log=lambda *args: None)
        probe.record(torch.tensor(0.), {'correction/test/min': rank + 1.,
            'correction/test/max': rank + 3., 'rank_mean': rank + 10.}, 0., torch.tensor(0.), 0.)
        row = json.loads((Path(directory) / 'train.jsonl').read_text().splitlines()[-1])
        assert row['correction/test/min'] == 1 and row['correction/test/max'] == 4
        assert row['rank_mean'] == 10.5
        # Actual hourly post_window: 23 no-grad DDP forwards followed by one
        # differentiable, checkpointed DDP forward, repeated across day updates.
        from neuralpom.v1.engine import post_window
        class HourlyRuntime:
            interval_seconds = 3600

            def __init__(self):
                self.core = SimpleNamespace(time=1000 + 1 / 24,
                                            x=torch.tensor([[float(rank + 1)]]))
                self.modes = []

            def advance(self, differentiable=False):
                assert not differentiable and not torch.is_grad_enabled()
                self.core.x = self.core.x * .99
                self.core.time += 1 / 24

            def correct(self, net):
                self.modes.append(torch.is_grad_enabled())
                self.core.x = self.core.x + net(self.core.x)

            def detach(self):
                self.core.x = self.core.x.detach()

            def score(self, *args):
                loss = self.core.x.square().mean()
                return loss, {'mse': loss}

        hourly = DDP(CheckpointedLinear(), broadcast_buffers=False, gradient_as_bucket_view=True)
        hourly_optimizer = torch.optim.AdamW(hourly.parameters(), lr=.001)
        runtime = HourlyRuntime()
        for window in range(2):
            hourly_optimizer.zero_grad(set_to_none=True)
            initial = runtime.core.x.detach().clone()
            weight = hourly.module.linear.weight.detach().clone()
            with torch.no_grad():
                z = initial
                for hour in range(24):
                    if window or hour:
                        z = z * .99
                    if hour < 23:
                        z = z + weight * z
                expected_grad = 2 * (1 + weight) * z.square()
                dist.all_reduce(expected_grad)
                expected_grad /= 2
            loss, _ = post_window(runtime, hourly, 0, 1000, window, 10)
            loss.backward()
            torch.testing.assert_close(hourly.module.linear.weight.grad, expected_grad)
            hourly_optimizer.step()
            runtime.detach()
        assert runtime.modes == ([False] * 23 + [True]) * 2
        Path(directory, f'rank_{rank}_passed').write_text('passed')
    finally:
        dist.destroy_process_group()


def test_ddp_multiple_checkpointed_forwards_and_restart(tmp_path):
    mp.spawn(_rank, args=(str(tmp_path / 'init'), str(tmp_path)), nprocs=2, join=True)
    assert len(list(tmp_path.glob('rank_*_passed'))) == 2
