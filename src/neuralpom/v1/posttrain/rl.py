"""Ten-day, on-policy full-field Gaussian RLOO posttraining.

Rollout collection owns physical state. Optimization owns only a detached
observation/action journal; no physical graph is retained or reconstructed.
"""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from neuralpom.models.corrector_net import CorrectorNet
from ..checkpoint import RestartStore, atomic_json, atomic_torch, cpu_tree, distributed, unwrap
from ..config import training_settings
from ..optimization import gradient_norm, clip_gradients
from ..execution import rank_zero_action, collective_call as collective
from ..sampling import stage_samples
from .policy import GaussianPolicy, validate_settings, batch_contexts, returns_to_go, leave_one_out, early_stop_reached
from .rollout import collect


class RLPosttrain:
    def __init__(self, config, runtime, logger, parent=None, resume=True):
        self.config, self.runtime, self.logger = config, runtime, logger
        self.rank = dist.get_rank() if distributed() else 0
        self.world = dist.get_world_size() if distributed() else 1
        self.options = config['stages']['posttrain']['rl']
        validate_settings(self.options, self.world, config['stages']['posttrain']['rollout_days'])
        if self.world != config['world_size']:
            raise ValueError('Launch world size differs from configuration')
        self.cases = stage_samples(config, 'posttrain')
        self.validation_cases = stage_samples(config, 'posttrain', split='validation')
        if len(self.cases) % self.options['contexts_per_batch']:
            raise ValueError('RL sampler cannot silently pad initial contexts')
        collective(lambda: runtime.require_cache(self.cases, self.validation_cases))
        self.output = Path(config['run_root']) / 'posttrain'
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / 'COMPLETE.json').exists():
            raise FileExistsError('Completed RL run cannot be relaunched')
        model = CorrectorNet(101, 60, config['model']['hidden_dim']).to(runtime.device)
        model.checkpoint_blocks = True
        path = Path(parent) if parent else Path(config['run_root']) / 'frozen/midtrain4.pt'
        def load_parent():
            value = torch.load(path, map_location='cpu', weights_only=False)
            if value['contract'] != runtime.contract() or value['stage'] != 'midtrain4':
                raise ValueError('RL must start from its matching frozen Midtrain4')
            if value.get('supervision') != config['supervision']:
                raise ValueError('Parent supervision differs')
            model.load_state_dict(value['model'], strict=True)
            return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'stage': value['stage'], 'step': value['step']}
        self.parent = collective(load_parent)
        self.model = DistributedDataParallel(model,
            device_ids=[runtime.device.index] if runtime.device.type == 'cuda' else None,
            broadcast_buffers=False, gradient_as_bucket_view=True) if distributed() else model
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
            lr=config['optimizer']['posttrain_lr'], weight_decay=config['optimizer']['weight_decay'])
        self.policy = GaussianPolicy(self.options['noise_std'])
        self.contract = {**runtime.contract(), 'training_settings': training_settings(config),
                         'stage': 'posttrain', 'method': 'rl', 'parent_sha256': self.parent['sha256']}
        self.restart = RestartStore(self.output, self.contract)
        self.step, self.best_step, self.bad_checks = 0, 0, 0
        self.best, self.stopped, self.initialized = float('inf'), False, False
        attempt = [str(time.time_ns()) if self.rank == 0 else None]
        if distributed():
            dist.broadcast_object_list(attempt, src=0)
        self.attempt = attempt[0]
        noise_seed = config['seed'] + 100003 * (self.rank + 1)
        random.seed(noise_seed)
        np.random.seed(noise_seed)
        torch.manual_seed(noise_seed)
        if runtime.device.type == 'cuda':
            torch.cuda.manual_seed(noise_seed)
        if (self.output / 'resume.json').exists():
            if not resume:
                raise FileExistsError('A fresh RL run requires a new run_root')
            value, _ = self.restart.load(self.model, self.optimizer)
            self.step = value['step']
            for key in ('best', 'best_step', 'bad_checks', 'stopped', 'initialized'):
                setattr(self, key, value['extra'][key])
            if value['cursor'] != {'next_batch': self.step}:
                raise ValueError('RL restart cursor disagrees with optimizer step')
        rank_zero_action(lambda: atomic_json(self.output / 'manifest.json', {
            'contract': self.contract, 'parent': self.parent, 'train_contexts': len(self.cases),
            'validation_contexts': len(self.validation_cases), 'independent_noise_seed_rule': 'seed+100003*(rank+1)',
            'reward': '-daily_original_loss/10', 'discount': 1.,
            'training_correction': 'canonical', 'validation_correction': 'deterministic_mean_D75',
            'policy_fixed_within_rollout': True, 'physical_backpropagation': False,
            'max_trajectories': self.options['max_trajectories'],
            'early_stopping_enabled': self.options.get('patience') is not None,
            'budget_basis': self.options.get('budget_basis'),
            'actions_per_trajectory': 10 * 86400 // runtime.interval_seconds,
            'online_steps_per_successful_trajectory': (10 * 86400 - runtime.interval_seconds) // runtime.spec.physical_step_seconds}))

    def save(self):
        self.restart.save(self.model, self.optimizer, {'next_batch': self.step}, self.step,
            extra={key: getattr(self, key) for key in ('best', 'best_step', 'bad_checks', 'stopped', 'initialized')})

    def weights(self, path, selection, *, validation_loss=None):
        atomic_torch(path, {'model': cpu_tree(unwrap(self.model).state_dict()),
            'contract': self.runtime.contract(), 'stage': 'posttrain', 'method': 'rl',
            'step': self.step, 'parent': self.parent, 'supervision': self.config['supervision'],
            'posttrain_gradient_mode': 'score_function_no_physics_gradient',
            'selection': selection, 'validation_loss': validation_loss, 'best_validation_loss': self.best, 'rl_settings': self.options})

    def log(self, values):
        def publish():
            row = {'attempt': self.attempt, 'step': self.step, 'trajectories': self.step * self.world, **values}
            with (self.output / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            self.logger.log(row, step=self.step)
            print(json.dumps(row, allow_nan=False), flush=True)
        rank_zero_action(publish)

    def validate(self):
        """Select by deterministic deployed mean, including numerical failures."""
        unwrap(self.model).eval()
        def evaluate():
            total, failures, steps = 0., 0, 0
            for member, day in self.validation_cases[self.rank::self.world]:
                episode = collect(self.runtime, unwrap(self.model), member, day, density75=True,
                    store_actions=False, failure_daily_loss=self.options['failure_daily_loss'])
                total += float(episode.daily_losses.mean())
                failures += int(episode.failed)
                steps += episode.physical_steps
            return torch.tensor([total, failures, steps], dtype=torch.float64, device=self.runtime.device)
        totals = collective(evaluate)
        if distributed():
            dist.all_reduce(totals)
        loss = float(totals[0]) / len(self.validation_cases)
        self.latest_validation = loss
        improved = loss < self.best - self.options.get('min_delta', 0.)
        if improved:
            self.best, self.best_step, self.bad_checks = loss, self.step, 0
        elif self.initialized:
            self.bad_checks += 1
        self.initialized = True
        self.stopped = early_stop_reached(self.options, self.bad_checks)
        def publish():
            self.weights(self.output / 'validation_weights' / f'step_{self.step:08d}.pt', 'validation_candidate', validation_loss=loss)
            if improved:
                self.weights(Path(self.config['run_root']) / 'frozen/posttrain_best.pt', 'deterministic_D75_10day_validation_best', validation_loss=loss)
        rank_zero_action(publish)
        self.log({'validation/deterministic_D75_10day_loss': loss,
                  'validation/numerical_failures': int(totals[1]),
                  'validation/physical_forward_steps': int(totals[2]),
                  'validation/best': self.best, 'validation/best_step': self.best_step,
                  'validation/bad_checks': self.bad_checks})
        unwrap(self.model).train()
        self.save()

    def optimize(self, episode, advantage):
        """Equal-rank trajectory average via DDP; sum joint scores over decisions."""
        opts = self.config['optimizer']
        lr = opts['posttrain_lr'] * min((self.step + 1) / max(1, opts['posttrain_warmup_updates']), 1.)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.optimizer.zero_grad(set_to_none=True)
        n = 10 * 86400 // self.runtime.interval_seconds
        surrogate = 0.
        dummy = torch.zeros(1, 101, self.config['grid']['im'], self.config['grid']['jm'], device=self.runtime.device)
        for index in range(n):
            # All ranks execute the same number of DDP forwards, even after failure.
            sync = self.model.no_sync() if distributed() and index < n - 1 else nullcontext()
            with sync:
                observation = episode.observations[index].to(self.runtime.device) if index < episode.valid_actions else dummy
                mean = self.model(observation)[0]
                if index < episode.valid_actions:
                    action = episode.actions[index].to(self.runtime.device)
                    loss = -advantage[index].detach() * self.policy.log_prob(mean, action)
                else:
                    loss = mean.sum(dtype=torch.float64) * 0.
                loss.backward()
                surrogate += float(loss.detach())
        norm = gradient_norm(self.model.parameters())
        finite = torch.isfinite(norm).to(torch.int32)
        if distributed():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError('Nonfinite RL parameter gradient')
        clip_gradients(self.model.parameters(), norm, opts['clip_norm'])
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        return lr, float(norm), surrogate

    def train(self):
        if not self.initialized:
            self.validate()
        maximum = self.options['max_trajectories'] // self.world
        cadence = self.options['validation_every_trajectories'] // self.world
        while self.step < maximum and not self.stopped:
            begin = time.monotonic()
            contexts = batch_contexts(self.cases, self.step, self.options['contexts_per_batch'], self.config['seed'])
            member, day = contexts[self.rank // self.options['group_size']]
            unwrap(self.model).train()
            episode = collective(lambda: collect(self.runtime, unwrap(self.model), member, day,
                policy=self.policy, failure_daily_loss=self.options['failure_daily_loss']))
            journal = {'attempt': self.attempt, 'update_before': self.step, 'rank': self.rank, 'member': member, 'start_day': day,
                'daily_losses': episode.daily_losses.cpu().tolist(), 'failed': episode.failed,
                'failure_lead': episode.failure_lead, 'failure_reason': episode.failure_reason,
                'valid_actions': episode.valid_actions, 'physical_steps': episode.physical_steps,
                'correction_diagnostics': episode.diagnostics}
            def write_episode():
                with (self.output / f'trajectories_rank{self.rank}.jsonl').open('a') as stream:
                    stream.write(json.dumps(journal, allow_nan=False) + '\n')
            collective(write_episode)
            local_returns = returns_to_go(-episode.daily_losses / 10., self.runtime.interval_seconds)
            gathered = [torch.empty_like(local_returns) for _ in range(self.world)]
            if distributed():
                dist.all_gather(gathered, local_returns)
            else:
                gathered[0] = local_returns
            advantage = leave_one_out(torch.stack(gathered), self.options['group_size'])[self.rank]
            lr, norm, surrogate = self.optimize(episode, advantage)
            totals = torch.tensor([float(episode.daily_losses.mean()), episode.failed,
                episode.physical_steps, episode.valid_actions, float(advantage.square().mean()), surrogate],
                dtype=torch.float64, device=self.runtime.device)
            if distributed():
                dist.all_reduce(totals)
            self.log({'train/mean_daily_loss': float(totals[0]) / self.world,
                'train/numerical_failures': int(totals[1]),
                'train/physical_forward_steps': int(totals[2]),
                'train/valid_actions': int(totals[3]), 'train/advantage_rms': (float(totals[4]) / self.world)**.5,
                'train/score_surrogate': float(totals[5]) / self.world,
                'train/gradient_norm_before_clip': norm, 'train/lr': lr,
                'train/equivalent_legacy_epoch': self.step / self.options.get('budget_basis', {}).get('ddp_trajectories_per_epoch', self.world) * self.world,
                'train/batch_seconds': time.monotonic() - begin})
            if self.step % cadence == 0 or self.step == maximum:
                self.validate()
            elif self.step == 1 or self.step % self.config['stages']['posttrain']['checkpoint_every_updates'] == 0:
                self.save()
        self.save()
        def finish():
            self.weights(Path(self.config['run_root']) / 'frozen/posttrain_last.pt', 'last_completed_update')
            self.weights(Path(self.config['run_root']) / 'frozen/posttrain.pt',
                'full_budget_final' if self.step == maximum else 'last_completed_update',
                validation_loss=getattr(self, 'latest_validation', None))
            atomic_json(self.output / 'COMPLETE.json', {'method': 'rl', 'step': self.step,
                'trajectories': self.step * self.world, 'early_stopped': self.stopped,
                'best_validation': self.best, 'best_step': self.best_step,
                'selected_weights': str(Path(self.config['run_root']) / 'frozen/posttrain.pt'),
                'selection': 'full_budget_final' if self.step == maximum else 'last_completed_update',
                'completed_budget': self.step == maximum, 'maximum_updates': maximum,
                'best_weights': str(Path(self.config['run_root']) / 'frozen/posttrain_best.pt'),
                'last_weights': str(Path(self.config['run_root']) / 'frozen/posttrain_last.pt')})
            self.logger.artifact(Path(self.config['run_root']) / 'frozen/posttrain.pt', 'rl-posttrain-final', 'model')
        rank_zero_action(finish)
