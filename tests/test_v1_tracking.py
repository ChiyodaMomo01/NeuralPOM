"""Source provenance captures the current checkout, including uncommitted code."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from neuralpom.v1.tracking import ExperimentLogger, git_source, source_files


def write(root, name, text='current source'):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_source_selection_excludes_archives_outputs_credentials_and_links(tmp_path):
    included = ['src/neuralpom/main.py', 'scripts/run.sh', 'scripts/validation/configs/test.json',
                'configs/v1/1h.json', 'pyproject.toml']
    excluded = ['archive/old.py', 'outputs/report.json', 'src/__pycache__/main.py',
                'scripts/outputs/result.json', 'configs/credentials.json', 'configs/api_key.json',
                'configs/token.yaml', 'configs/.env', 'configs/secrets/auth.json',
                'src/.hidden/extra.py', 'src/weights.pt']
    for name in included + excluded:
        write(tmp_path, name)
    external = write(tmp_path, 'outside/private.py')
    (tmp_path / 'src/link.py').symlink_to(external)
    (tmp_path / 'src/linked_directory').symlink_to(external.parent, target_is_directory=True)
    assert {str(path.relative_to(tmp_path)) for path in source_files(tmp_path)} == set(included)


def test_git_baseline_is_distinct_from_current_dirty_snapshot(tmp_path):
    def git(*args):
        return subprocess.run(['git', '-C', str(tmp_path), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    git('init')
    path = write(tmp_path, 'src/model.py', 'baseline')
    git('add', '.')
    git('-c', 'user.name=NeuralPOM Test', '-c', 'user.email=test@example.invalid',
        '-c', 'commit.gpgsign=false', 'commit', '-m', 'fixture')
    baseline = git('rev-parse', 'HEAD')
    assert git_source(tmp_path)['git_dirty'] is False
    path.write_text('dirty implementation')
    write(tmp_path, 'configs/untracked.json', '{}')
    logger = ExperimentLogger.__new__(ExperimentLogger)
    logger.metadata = {'id': 'source-fixture'}
    logger.run = MagicMock(summary={})
    artifact = MagicMock()
    captured = {}
    artifact.add_file.side_effect = lambda path, name: captured.update({name: Path(path).read_text()})
    logger.wandb = SimpleNamespace(Artifact=MagicMock(return_value=artifact))
    logger._record_source(tmp_path)
    provenance = logger.metadata['code_source']
    assert provenance['git_head'] == baseline
    assert provenance['git_dirty'] is True
    assert provenance['git_branch']
    assert captured == {'src/model.py': 'dirty implementation', 'configs/untracked.json': '{}'}
    assert logger.wandb.Artifact.call_args.kwargs['type'] == 'source-code'
    logger.run.log_artifact.assert_called_once_with(artifact)
    assert logger.run.summary['code_source'] == provenance


def test_no_git_installation_still_records_offline_code_artifact(tmp_path):
    source = tmp_path / 'source'
    write(source, 'src/neuralpom/v1/tracking.py')
    write(source, 'pyproject.toml')
    run = MagicMock(url=None, summary={})
    wandb = SimpleNamespace(init=MagicMock(return_value=run), Settings=MagicMock(), Artifact=MagicMock())
    config = {'run_root': str(tmp_path / 'run'), 'wandb': {'project': 'test', 'mode': 'offline'}}
    with patch.dict(sys.modules, wandb=wandb), \
         patch('neuralpom.v1.tracking.__file__', str(source / 'src/neuralpom/v1/tracking.py')), \
         patch('neuralpom.v1.tracking.subprocess.run', side_effect=FileNotFoundError('git')):
        logger = ExperimentLogger(config, 'pretrain', rank=0)
        logger.finish()
    provenance = json.loads((logger.directory / 'run.json').read_text())['code_source']
    assert provenance['git_status'] == 'unavailable'
    assert provenance['git_head'] is None and provenance['git_dirty'] is None
    assert provenance['snapshot_files'] == ['pyproject.toml', 'src/neuralpom/v1/tracking.py']
    run.log_artifact.assert_called_once()
    assert wandb.init.call_args.kwargs['mode'] == 'offline'


def test_nonzero_rank_never_reads_source_or_initializes_wandb(tmp_path):
    with patch('neuralpom.v1.tracking.git_source') as git, \
         patch('neuralpom.v1.tracking.source_files') as files:
        logger = ExperimentLogger({'run_root': str(tmp_path)}, 'pretrain', rank=1)
    git.assert_not_called()
    files.assert_not_called()
    assert logger.run is None


def test_installed_package_inside_git_checkout_uses_only_package_snapshot(tmp_path, monkeypatch):
    subprocess.run(['git', 'init', str(tmp_path)], check=True, capture_output=True)
    write(tmp_path, 'pyproject.toml', 'unrelated outer checkout')
    installed = tmp_path / '.venv/lib/python3.10/site-packages'
    tracking = write(installed, 'neuralpom/v1/tracking.py', 'installed implementation')
    write(installed, 'neuralpom/__init__.py')
    write(installed, 'neuralpom/__pycache__/tracking.py')
    write(installed, 'neuralpom/credentials.json')
    write(installed, 'another_package/private.py')
    write(installed, 'pyproject.toml', 'unrelated installation data')
    write(installed, 'neuralpom-1.0.0.dist-info/METADATA', 'Name: neuralpom\nVersion: 1.0.0\n')
    monkeypatch.syspath_prepend(str(installed))
    run = MagicMock(url=None, summary={})
    artifact, captured = MagicMock(), {}
    artifact.add_file.side_effect = lambda path, name: captured.update({name: Path(path).read_text()})
    wandb = SimpleNamespace(init=MagicMock(return_value=run), Settings=MagicMock(),
                            Artifact=MagicMock(return_value=artifact))
    config = {'run_root': str(tmp_path / 'run'), 'wandb': {'project': 'test', 'mode': 'offline'}}
    with patch.dict(sys.modules, wandb=wandb), \
         patch('neuralpom.v1.tracking.__file__', str(tracking)), \
         patch('neuralpom.v1.tracking.subprocess.run') as git:
        logger = ExperimentLogger(config, 'pretrain', rank=0)
        logger.finish()
    git.assert_not_called()
    record = json.loads((logger.directory / 'run.json').read_text())
    provenance = record['code_source']
    assert record['execution_complete'] is True
    assert provenance['snapshot_scope'] == 'installed package'
    assert provenance['package_version'] == '1.0.0'
    assert provenance['git_status'] == 'not_applicable'
    assert provenance['git_head_role'] == 'not_applicable'
    assert all(provenance[key] is None for key in ('git_head', 'git_branch', 'git_dirty'))
    assert set(captured) == {'neuralpom/__init__.py', 'neuralpom/v1/tracking.py'}
    assert captured['neuralpom/v1/tracking.py'] == 'installed implementation'
    run.log_artifact.assert_called_once_with(artifact)


def test_online_initialization_failure_has_no_fallback(tmp_path):
    wandb = SimpleNamespace(init=MagicMock(side_effect=RuntimeError('online failure')), Settings=MagicMock())
    with patch.dict(sys.modules, wandb=wandb), pytest.raises(RuntimeError, match='online failure'):
        ExperimentLogger({'run_root': str(tmp_path), 'wandb': {'project': 'test', 'mode': 'online'}},
                         'pretrain', rank=0)
    assert wandb.init.call_count == 1


def test_snapshot_failure_finishes_run_and_propagates(tmp_path):
    run = MagicMock(url=None, summary={})
    wandb = SimpleNamespace(init=MagicMock(return_value=run), Settings=MagicMock())
    with patch.dict(sys.modules, wandb=wandb), \
         patch.object(ExperimentLogger, '_record_source', side_effect=OSError('source unreadable')), \
         pytest.raises(OSError, match='source unreadable'):
        ExperimentLogger({'run_root': str(tmp_path), 'wandb': {'project': 'test', 'mode': 'offline'}},
                         'pretrain', rank=0)
    run.finish.assert_called_once_with(exit_code=1)
