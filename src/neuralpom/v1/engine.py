"""Offline pretraining, full-BPTT curriculum and continuous-state TBPTT."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from neuralpom.models.corrector_net import CorrectorNet
from .checkpoint import RestartStore, atomic_json, atomic_torch, cpu_tree, distributed, unwrap
from .config import training_settings
from .execution import rank_zero_action
from .optimization import gradient_norm, clip_gradients
from .sampling import stage_samples, stage_plan, post_windows
from .posttrain.statistics import correction_statistics
from .posttrain.truncated import post_window


PARENTS = {'midtrain4': 'pretrain', 'posttrain': 'midtrain4'}


def read_records(path):
    path = Path(path)
    if not path.exists():
        return []
    lines, records = path.read_text().splitlines(), []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            break
    return records


def repair_incomplete_tail(path):
    """Keep every completed journal entry when a process died during append."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    final = data.splitlines(keepends=True)[-1]
    try:
        json.loads(final)
    except json.JSONDecodeError:
        with path.open('r+b') as stream:
            stream.truncate(len(data) - len(final))
    else:
        if not data.endswith(b'\n'):
            with path.open('ab') as stream:
                stream.write(b'\n')


def reconcile_resource_ledger(directory, progress_step=None):
    """Migrate old timing records, then rebuild the disposable progress copy.

    Attempt zero denotes legacy records whose attempt cannot be recovered.
    Completed resource events are immutable and identified by (attempt, step).
    """
    directory = Path(directory)
    timing_path, ledger_path = directory / 'timing.jsonl', directory / 'resource_usage.jsonl'
    timing, ledger = read_records(timing_path), read_records(ledger_path)
    attempts = read_records(directory / 'attempts.jsonl')
    repair_incomplete_tail(ledger_path)
    seen = {(row['attempt'], row['step']) for row in ledger}
    additions = []
    for row in timing:
        if 'attempt' not in row and any(all(event.get(key) == value for key, value in row.items()) for event in ledger):
            continue
        inferred_attempt = 0
        for attempt in attempts:
            if row['step'] > attempt.get('resume_from_step', 0):
                inferred_attempt = attempt['attempt']
        event = {'attempt': row.get('attempt', inferred_attempt), **row}
        if 'attempt' not in row:
            event['origin'] = 'migrated_legacy_timing'
        identity = (event['attempt'], event['step'])
        if identity not in seen:
            additions.append(event)
            ledger.append(event)
            seen.add(identity)
    for attempt in attempts:
        # Old progress logs discarded replayed rows but retained their measured
        # cost in the next attempt record. Add only the part absent from events.
        identity = (attempt['attempt'], -1)
        if identity in seen:
            continue
        covered = [event for event in ledger if event['attempt'] == attempt['attempt'] - 1
                   and event['step'] > attempt.get('resume_from_step', 0)]
        hours = max(0., attempt.get('discarded_metrics_gpu_hours', 0.) -
                    sum(event.get('allocated_gpu_hours', 0.) for event in covered))
        steps = max(0, attempt.get('discarded_metrics_physics_steps', 0) -
                    sum(event.get('online_forward_physics_steps_all_ranks', 0) for event in covered))
        if hours > 1e-15 or steps:
            event = {'attempt': attempt['attempt'], 'step': -1,
                'origin': 'migrated_legacy_replay_summary', 'allocated_gpu_hours': hours,
                'online_forward_physics_steps_all_ranks': steps}
            additions.append(event)
            ledger.append(event)
            seen.add(identity)
    if additions:
        with ledger_path.open('a') as stream:
            for event in additions:
                stream.write(json.dumps(event, allow_nan=False) + '\n')
    latest = {}
    for event in ledger:
        if event['step'] < 0:  # Historical cost aggregate, not optimizer progress.
            continue
        previous = latest.get(event['step'])
        if previous is None or event['attempt'] >= previous['attempt']:
            latest[event['step']] = {key: value for key, value in event.items() if key != 'origin'}
    if ledger or timing_path.exists():
        temporary = timing_path.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(latest[step], allow_nan=False) + '\n' for step in sorted(latest) if progress_step is None or step <= progress_step))
        temporary.replace(timing_path)


def trim_records(path, saved_step):
    """Discard progress metrics for updates that will be replayed after restart."""
    path = Path(path)
    if not path.exists():
        return saved_step
    rows = read_records(path)
    highest = max([saved_step] + [int(row['step']) for row in rows])
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows if row['step'] <= saved_step))
    temporary.replace(path)
    return highest


def validate_cursor(cursor, step, plan, windows):
    epoch, batch, window = (cursor[key] for key in ('epoch', 'batch', 'window'))
    batches = plan['trajectory_batches_per_epoch']
    if not (0 <= epoch <= plan['epochs'] and 0 <= batch <= batches and 0 <= window < windows):
        raise ValueError('Restart cursor is outside the registered epoch budget')
    if (epoch == plan['epochs'] and (batch or window)) or (batch == batches and window):
        raise ValueError('Restart cursor crosses a dataset boundary')
    if step != (epoch * batches + batch) * windows + window:
        raise ValueError('Restart optimizer step and sampler cursor disagree')


def full_bptt(runtime, model, member, day, steps, training=True):
    """Full graph; mean of corrected endpoint losses against reference fields."""
    record = runtime.load_cached(member, day)
    losses, metrics, corrections = [], {}, []
    for index in range(1, steps + 1):
        lead = index * runtime.interval_seconds
        if index > 1:
            runtime.advance(differentiable=training)
        corrections.append(runtime.correct(model, record.get('packed') if index == 1 else None) or {})
        post, post_metrics = runtime.score(member, day, lead, include_eta=steps > 1)
        losses.append(post)
        metrics.update({f'{index}/post/{key}': value.detach() for key, value in post_metrics.items()})
    metrics.update(correction_statistics(corrections))
    return torch.stack(losses).mean(), metrics


class Training:
    def __init__(self, config, stage, runtime, logger, parent=None, resume=True):
        self.config, self.stage, self.runtime, self.logger = config, stage, runtime, logger
        self.rank = dist.get_rank() if distributed() else 0
        self.world = dist.get_world_size() if distributed() else 1
        if self.world != config['world_size']:
            raise ValueError('Launch world size differs from registered configuration')
        self.plan = stage_plan(config, stage)
        category = 'midtrain' if stage.startswith('midtrain') else stage
        self.checkpoint_interval = config['stages'][category].get('checkpoint_every_updates', 1)
        runtime.require_cache(stage_samples(config, stage),
                              stage_samples(config, stage, split='validation') if stage == 'pretrain' else ())
        self.output = Path(config['run_root']) / stage
        self.output.mkdir(parents=True, exist_ok=True)
        model = CorrectorNet(101, 60, config['model']['hidden_dim']).to(runtime.device)
        model.checkpoint_blocks = True
        self.model = DistributedDataParallel(model, device_ids=[runtime.device.index] if runtime.device.type == 'cuda' else None,
                         broadcast_buffers=False, gradient_as_bucket_view=True) if distributed() else model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr,
                                          weight_decay=config['optimizer']['weight_decay'])
        self.contract = {**runtime.contract(), 'training_settings': training_settings(config), 'stage': stage}
        self.restart = RestartStore(self.output, self.contract)
        self.cursor = {'epoch': 0, 'batch': 0, 'window': 0}
        self.step = 0
        self.best = float('inf')
        self.bad_epochs = 0
        self.parent = None
        self.physical = None
        self.stopped = False
        resumed = resume and (self.output / 'resume.json').exists()
        if not resume and (self.output / 'resume.json').exists():
            raise FileExistsError('A fresh run requires a new run_root')
        if resume and (self.output / 'resume.json').exists():
            value, self.physical = self.restart.load(self.model, self.optimizer)
            self.cursor, self.step = value['cursor'], value['step']
            extra = value['extra']
            self.best, self.bad_epochs = extra['best'], extra['bad_epochs']
            self.parent, self.stopped = extra['parent'], extra['stopped']
        elif stage in PARENTS:
            path = Path(parent) if parent else Path(config['run_root']) / 'frozen' / (PARENTS[stage] + '.pt')
            value = torch.load(path, map_location='cpu', weights_only=False)
            if value['contract'] != runtime.contract() or value['stage'] != PARENTS[stage]:
                raise ValueError('Parent weights have incompatible state contract or stage')
            if value['stage'] != 'pretrain' and value.get('supervision') != self.config['supervision']:
                raise ValueError('Parent weights use a different supervision objective')
            unwrap(self.model).load_state_dict(value['model'], strict=True)
            self.parent = {'path': str(path), 'stage': value['stage']}
        elif parent is not None:
            raise ValueError('Pretrain starts from random backbone and zero output head')
        windows = post_windows(config) if stage == 'posttrain' else 1
        validate_cursor(self.cursor, self.step, self.plan, windows)
        rank_zero_action(lambda: self.record_attempt(resumed))
        rank_zero_action(lambda: atomic_json(self.output / 'manifest.json', {'contract': self.contract,
                'plan': self.plan, 'parent': self.parent, 'cursor': self.cursor,
                'metric_aggregation': 'Per-trajectory rank means, except correction min/max are global rank extrema'}))

    def record_attempt(self, resumed):
        reconcile_resource_ledger(self.output)
        discarded = []
        timing_path = self.output / 'timing.jsonl'
        if resumed and timing_path.exists():
            discarded = [row for row in read_records(timing_path) if row['step'] > self.step]
        repair_incomplete_tail(self.output / 'resource_usage.jsonl')
        highest = trim_records(self.output / 'train.jsonl', self.step) if resumed else self.step
        if resumed:
            trim_records(self.output / 'timing.jsonl', self.step)
        path = self.output / 'attempts.jsonl'
        attempts = read_records(path)
        repair_incomplete_tail(path)
        self.attempt = len(attempts) + 1
        event = {'attempt': self.attempt, 'resume_from_step': self.step,
                 'replayed_updates': max(0, highest - self.step), 'resumed': bool(resumed),
                 'discarded_metrics_gpu_hours': sum(row.get('allocated_gpu_hours', 0) for row in discarded),
                 'discarded_metrics_physics_steps': sum(row.get('online_forward_physics_steps_all_ranks', 0) for row in discarded)}
        with path.open('a') as stream:
            stream.write(json.dumps(event) + '\n')
        self.logger.log({'restart/' + key: value for key, value in event.items()}, self.step)

    @property
    def base_lr(self):
        category = 'midtrain' if self.stage.startswith('midtrain') else self.stage
        return self.config['optimizer'][category + '_lr']

    def set_lr(self):
        opts = self.config['optimizer']
        update = self.step + 1
        lr = self.base_lr
        if self.stage == 'pretrain':
            warm = min(opts['warmup_epochs'] * self.plan['updates_per_epoch'], self.plan['maximum_updates'])
            if warm and update <= warm:
                lr *= update / warm
            else:
                fraction = (update - warm) / max(1, self.plan['maximum_updates'] - warm)
                lr = opts['pretrain_final_lr'] + .5 * (lr - opts['pretrain_final_lr']) * (1 + math.cos(math.pi * fraction))
        elif self.stage == 'posttrain':
            lr *= min(update / max(1, opts['posttrain_warmup_updates']), 1)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return lr

    def optimize(self, loss):
        finite = torch.isfinite(loss.detach()).to(torch.int32)
        if distributed():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError('Nonfinite objective')
        loss.backward()
        norm = gradient_norm(self.model.parameters())
        finite = torch.isfinite(norm).to(torch.int32)
        if distributed():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError('Nonfinite parameter gradient')
        clip_gradients(self.model.parameters(), norm, self.config['optimizer']['clip_norm'])
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        self.runtime.detach()
        return norm.detach()

    def save(self, physical=False):
        begin = time.monotonic()
        self.restart.save(self.model, self.optimizer, dict(self.cursor), self.step,
            self.runtime.snapshot() if physical else None,
            {'best': self.best, 'bad_epochs': self.bad_epochs, 'parent': self.parent, 'stopped': self.stopped})
        self.last_checkpoint_seconds = time.monotonic() - begin

    def freeze(self):
        def publish():
            path = Path(self.config['run_root']) / 'frozen' / (self.stage + '.pt')
            atomic_torch(path, {'model': cpu_tree(unwrap(self.model).state_dict()),
                'contract': self.runtime.contract(), 'stage': self.stage, 'step': self.step,
                'parent': self.parent, 'supervision': self.config['supervision'],
                'posttrain_gradient_mode': self.config['stages']['posttrain']['gradient_mode'],
                'selection': 'validation_best' if self.stage == 'pretrain' else 'final'})
            self.logger.artifact(path, self.stage + '-weights', 'model')
        rank_zero_action(publish)

    def record(self, loss, metrics, lr, norm, elapsed):
        values = {'loss': loss.detach(), 'grad_norm': norm, 'lr': lr, 'step_seconds': elapsed, **metrics}
        names = list(values)
        tensor = torch.stack([
            values[key].detach().to(device=self.runtime.device, dtype=torch.float64).reshape(())
            if torch.is_tensor(values[key]) else torch.tensor(values[key], dtype=torch.float64, device=self.runtime.device)
            for key in names])
        if distributed():
            minima = [index for index, name in enumerate(names) if name.startswith('correction/') and name.endswith('/min')]
            maxima = [index for index, name in enumerate(names) if name.startswith('correction/') and name.endswith('/max')]
            minimum, maximum = tensor[minima].clone(), tensor[maxima].clone()
            dist.all_reduce(tensor)
            tensor /= self.world
            if minima:
                dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
                tensor[minima] = minimum
            if maxima:
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
                tensor[maxima] = maximum
        row = dict(zip(names, tensor.cpu().tolist()))
        def publish():
            row.update(step=self.step, **self.cursor)
            with (self.output / 'train.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            self.logger.log({'train/' + key: value for key, value in row.items()}, self.step)
        rank_zero_action(publish)

    def record_diagnostics(self, member, day, epoch):
        def publish():
            row = self.runtime.training_diagnostics(member, day)
            row.update(step=self.step + 1, epoch=epoch, rank=self.rank,
                       sample_policy='rank0 first trajectory first window of each posttrain epoch',
                       endpoint_phase='post_correction')
            path = self.output / 'diagnostics' / f'epoch_{epoch:04d}.json'
            atomic_json(path, row)
            self.logger.log({'diagnostics/' + key: value for key, value in row['metrics'].items()}, self.step + 1)
            self.logger.artifact(path, f'{self.stage}-epoch-{epoch}-diagnostics', 'diagnostics')
        rank_zero_action(publish)

    def record_timing(self, compute_seconds, checkpoint_seconds, iteration_seconds,
                      physical_steps=0, diagnostics_seconds=0., epoch_finalization_seconds=0.):
        cuda = self.runtime.device.type == 'cuda'
        seconds = torch.tensor([compute_seconds, checkpoint_seconds, iteration_seconds, diagnostics_seconds, epoch_finalization_seconds,
                                torch.cuda.max_memory_allocated(self.runtime.device) if cuda else 0,
                                torch.cuda.max_memory_reserved(self.runtime.device) if cuda else 0],
                               dtype=torch.float64, device=self.runtime.device)
        totals = torch.tensor([physical_steps, iteration_seconds / 3600 if cuda else 0.],
                              dtype=torch.float64, device=self.runtime.device)
        if distributed():
            dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
            dist.all_reduce(totals)
        values = dict(zip(('compute_seconds', 'checkpoint_seconds', 'iteration_seconds', 'diagnostics_seconds', 'epoch_finalization_seconds',
                           'peak_allocated_bytes', 'peak_reserved_bytes'), seconds.cpu().tolist()))
        values.update(online_forward_physics_steps_all_ranks=int(totals[0]), allocated_gpu_hours=float(totals[1]))
        def publish():
            event = {'attempt': self.attempt, 'step': self.step, **values}
            with (self.output / 'resource_usage.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False) + '\n')
            with (self.output / 'timing.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False) + '\n')
            self.logger.log({'timing/attempt': self.attempt, **{'timing/' + key: value for key, value in values.items()}}, self.step)
        rank_zero_action(publish)

    def resource_summary(self):
        reconcile_resource_ledger(self.output, self.step)
        def read_rows(name):
            path = self.output / name
            return read_records(path)
        rows = read_rows('resource_usage.jsonl')
        attempts = []
        if not rows:  # Older checkpoints predate the append-only resource ledger.
            rows, attempts = read_rows('timing.jsonl'), read_rows('attempts.jsonl')
        summary = {
            'allocated_gpu_hours': sum(row.get('allocated_gpu_hours', 0) for row in rows) + sum(row.get('discarded_metrics_gpu_hours', 0) for row in attempts),
            'online_forward_physics_steps_all_ranks': sum(row.get('online_forward_physics_steps_all_ranks', 0) for row in rows) + sum(row.get('discarded_metrics_physics_steps', 0) for row in attempts),
            'peak_allocated_bytes': max((row.get('peak_allocated_bytes', 0) for row in rows), default=0),
            'legacy_records_without_gpu_measurement': sum('allocated_gpu_hours' not in row for row in rows),
            'accounting': 'Completed iteration wall time summed over ranks; includes diagnostics, checkpoint I/O and epoch finalization. Replayed measured updates remain counted. Preparation, startup and unmeasured interrupted work excluded.',
            'physics_step_accounting': 'Completed online forward integration steps of 600 seconds, summed over ranks. Cached initial endpoint and activation-checkpoint recomputation are excluded; recomputation cost remains in wall time.'}
        atomic_json(self.output / 'resources.json', summary)
        self.logger.log({'resources/' + key: value for key, value in summary.items() if not isinstance(value, str)}, self.step)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        totals = torch.zeros(2, dtype=torch.float64, device=self.runtime.device)
        cases = stage_samples(self.config, self.stage, split='validation')
        for member, day in cases[self.rank::self.world]:
            loss, _ = full_bptt(self.runtime, unwrap(self.model), member, day,
                                self.plan['rollout_steps'], training=False)
            totals[0] += loss.double()
            totals[1] += 1
            self.runtime.detach()
        if distributed():
            dist.all_reduce(totals)
        if totals[1] == 0:
            raise ValueError('Validation set is empty')
        value = float(totals[0] / totals[1])
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite validation loss')
        rank_zero_action(lambda: self.logger.log({'validation/loss': value, 'validation/unique_samples': float(totals[1])}, self.step))
        self.model.train()
        return value

    def train(self):
        cases = stage_samples(self.config, self.stage)
        sampler = DistributedSampler(cases, num_replicas=self.world, rank=self.rank,
                                     shuffle=True, seed=self.config['seed'], drop_last=False)
        first = dict(self.cursor)
        for epoch in range(first['epoch'], self.plan['epochs']):
            if self.stopped:
                break
            sampler.set_epoch(epoch)
            order = list(sampler)
            self.model.train()
            last_iteration = None
            for batch, index in enumerate(order):
                if epoch == first['epoch'] and batch < first['batch']:
                    continue
                member, day = cases[index]
                start_window = first['window'] if epoch == first['epoch'] and batch == first['batch'] else 0
                windows = post_windows(self.config) if self.stage == 'posttrain' else 1
                if self.stage == 'posttrain':
                    if start_window:
                        if self.physical is None:
                            raise ValueError('Partial trajectory restart lacks full physical state')
                        self.runtime.restore(self.physical)
                        self.physical = None
                    else:
                        self.runtime.load_cached(member, day)
                for window in range(start_window, windows):
                    begin = time.monotonic()
                    forward_steps = getattr(self.runtime, 'physical_forward_steps', 0)
                    if self.runtime.device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(self.runtime.device)
                    self.optimizer.zero_grad(set_to_none=True)
                    lr = self.set_lr()
                    if self.stage == 'posttrain':
                        loss, metrics = post_window(self.runtime, self.model, member, day, window,
                                                    self.config['stages']['posttrain']['rollout_days'],
                                                    self.config['stages']['posttrain']['tbptt_hours'])
                    else:
                        loss, metrics = full_bptt(self.runtime, self.model, member, day, self.plan['rollout_steps'])
                    diagnostics_seconds = 0.
                    if self.stage == 'posttrain' and batch == 0 and window == 0 and hasattr(self.runtime, 'training_diagnostics'):
                        diagnostic_begin = time.monotonic()
                        self.record_diagnostics(member, day, epoch)
                        diagnostics_seconds = time.monotonic() - diagnostic_begin
                    norm = self.optimize(loss)
                    self.cursor = {'epoch': epoch, 'batch': batch, 'window': window + 1} if window + 1 < windows else {'epoch': epoch, 'batch': batch + 1, 'window': 0}
                    if self.runtime.device.type == 'cuda':
                        torch.cuda.synchronize(self.runtime.device)
                    compute_seconds = time.monotonic() - begin - diagnostics_seconds
                    physical_steps = getattr(self.runtime, 'physical_forward_steps', 0) - forward_steps
                    metrics['online_forward_physics_steps'] = physical_steps
                    self.record(loss, metrics, lr, norm, compute_seconds)
                    epoch_tail = batch + 1 == len(order) and window + 1 == windows
                    if epoch_tail:
                        last_iteration = (begin, compute_seconds, physical_steps, diagnostics_seconds)
                    else:
                        checkpoint_seconds = 0.
                        if self.step % self.checkpoint_interval == 0:
                            self.save(physical=self.stage == 'posttrain' and window + 1 < windows)
                            checkpoint_seconds = self.last_checkpoint_seconds
                        self.record_timing(compute_seconds, checkpoint_seconds, time.monotonic() - begin,
                                           physical_steps, diagnostics_seconds)
                    del loss, metrics, norm
            epoch_finalize_begin = time.monotonic()
            if self.stage == 'pretrain':
                validation = self.validate()
                setting = self.config['stages']['pretrain']
                if validation < self.best - setting['min_delta']:
                    self.best, self.bad_epochs = validation, 0
                    self.freeze()
                else:
                    self.bad_epochs += 1
                self.stopped = self.bad_epochs >= setting['patience']
            if self.stage != 'pretrain' and epoch + 1 == self.plan['epochs']:
                self.freeze()
            self.cursor = {'epoch': epoch + 1, 'batch': 0, 'window': 0}
            self.save()
            if last_iteration is not None:
                begin, compute_seconds, physical_steps, diagnostics_seconds = last_iteration
                self.record_timing(compute_seconds, self.last_checkpoint_seconds, time.monotonic() - begin,
                                   physical_steps, diagnostics_seconds, time.monotonic() - epoch_finalize_begin)
        rank_zero_action(self.resource_summary)
        rank_zero_action(lambda: atomic_json(self.output / 'COMPLETE.json', {'stage': self.stage, 'step': self.step,
                'cursor': self.cursor, 'early_stopped': self.stopped,
                'best_validation': self.best if math.isfinite(self.best) else None}))
        return self.step
