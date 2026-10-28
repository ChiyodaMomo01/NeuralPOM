"""Required W&B experiment tracking with local, recoverable metrics."""
from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import subprocess
import time
import uuid


def source_files(root, *, installed=False):
    """Select active text source; never follow links or collect run data/secrets."""
    root = Path(root)
    excluded = {'archive', 'outputs', 'output', 'checkpoints', 'logs', 'tracking',
                'wandb', '__pycache__', 'credentials', 'secrets'}
    extensions = {'.py', '.sh', '.json', '.yaml', '.yml', '.toml'}
    selected = []
    for name in (('neuralpom',) if installed else ('src', 'scripts', 'configs')):
        directory = root / name
        if directory.is_symlink():
            continue
        for current, directories, files in os.walk(directory):
            directories[:] = sorted(item for item in directories
                                     if not item.startswith('.') and item.lower() not in excluded
                                     and not (Path(current) / item).is_symlink())
            for name in sorted(files):
                path = Path(current) / name
                lower = name.lower()
                if (path.is_symlink() or lower.startswith('.') or path.suffix.lower() not in extensions
                        or any(word in lower for word in ('credential', 'secret', 'api_key'))
                        or path.stem.lower() in {'token', 'tokens', 'apikey'}):
                    continue
                selected.append(path)
    project = root / 'pyproject.toml'
    if not installed and project.is_file() and not project.is_symlink():
        selected.append(project)
    return sorted(selected)


def git_source(root):
    """HEAD describes the Git baseline; the code artifact captures dirty files."""
    result = {'git_head': None, 'git_branch': None, 'git_dirty': None}
    def git(*args):
        return subprocess.run(['git', '-C', str(root), *args], check=True,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    try:
        result['git_head'] = git('rev-parse', 'HEAD')
        branch = git('rev-parse', '--abbrev-ref', 'HEAD')
        result['git_branch'] = None if branch == 'HEAD' else branch
        result['git_dirty'] = bool(git('status', '--porcelain', '--untracked-files=all'))
        result['git_status'] = 'available'
    except (OSError, subprocess.SubprocessError) as error:
        result['git_status'] = 'unavailable'
        result['git_error'] = type(error).__name__
    return result


def _scalar(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


class ExperimentLogger:
    """One W&B run per stage; only rank zero performs external I/O.

    Offline mode is explicit and retains the W&B spool for later ``wandb sync``.
    Online initialization errors propagate; there is no silent logging fallback.
    """

    def __init__(self, config, stage, rank=None, *, resume=False):
        self.rank = int(os.environ.get("RANK", "0")) if rank is None else rank
        self.run = None
        self.stage = stage
        self.config = config
        self.directory = Path(config["run_root"]) / stage / "tracking"
        self.wandb = None
        if self.rank:
            return
        import wandb
        self.wandb = wandb
        settings = config.get("wandb", {})
        if not settings.get("project") or settings.get("mode", "online") not in ("online", "offline"):
            raise ValueError("Experiment requires a W&B project and online/offline mode")
        self.directory.mkdir(parents=True, exist_ok=True)
        metadata = self.directory / "run.json"
        if metadata.exists():
            previous = json.loads(metadata.read_text())
            if not resume and previous.get("execution_complete"):
                raise FileExistsError(f"Completed experiment already exists: {metadata}")
            run_id = previous["id"]
        else:
            run_id = uuid.uuid4().hex[:12]
        self.run = wandb.init(
            entity=settings.get("entity"), project=settings["project"],
            group=settings.get("group", f"p0-fixed-{config.get('frequency', 'validation')}-seed{config.get('seed', 0)}"),
            job_type=stage, name=settings.get("name", f"{config.get('frequency', 'fixed')}/{stage}/seed{config.get('seed', 0)}"),
            id=run_id, resume="allow", mode=settings.get("mode", "online"),
            dir=str(self.directory), config=config,
            settings=wandb.Settings(init_timeout=90, save_code=False),
        )
        self.run.define_metric("progress/update")
        self.run.define_metric("*", step_metric="progress/update")
        self.metadata = {"id": run_id, "url": self.run.url, "project": settings["project"],
                         "entity": settings.get("entity"), "mode": settings.get("mode", "online"),
                         "stage": stage, "execution_complete": False}
        try:
            package = Path(__file__).resolve().parents[1]
            checkout = package.parent.name == 'src' and (package.parent.parent / 'pyproject.toml').is_file()
            self._record_source(package.parent.parent if checkout else package.parent,
                                installed=not checkout)
        except Exception:
            self.run.finish(exit_code=1)
            self.run = None
            raise
        self._write_metadata()
        (self.directory / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    def _record_source(self, root, *, installed=False):
        files = source_files(root, installed=installed)
        if not files:
            raise RuntimeError(f'No active source files found at {root}')
        if installed:
            # A wheel may live in another repository's .venv. That repository's
            # HEAD is not the installed package's source revision.
            source = dict(git_head=None, git_branch=None, git_dirty=None,
                          git_status='not_applicable', git_head_role='not_applicable',
                          snapshot_scope='installed package', package_name='neuralpom')
            try:
                source['package_version'] = version('neuralpom')
            except PackageNotFoundError:
                source['package_version'] = None
        else:
            source = {**git_source(root), 'git_head_role': 'baseline; current source is the artifact snapshot',
                      'snapshot_scope': 'active src/scripts/configs and pyproject.toml'}
        provenance = {**source,
                      'snapshot_files': [str(path.relative_to(root)) for path in files],
                      'snapshot_time': time.time(),
                      'artifact_name': f"source-{self.metadata['id']}"}
        artifact = self.wandb.Artifact(provenance['artifact_name'], type='source-code', metadata=provenance)
        for path in files:
            artifact.add_file(str(path), name=str(path.relative_to(root)))
        self.run.log_artifact(artifact)
        self.metadata['code_source'] = provenance
        self.run.summary['code_source'] = provenance

    def _write_metadata(self):
        path = self.directory / "run.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.metadata, indent=2) + "\n")
        temp.replace(path)

    def log(self, values, step=None):
        if self.rank:
            return
        values = {key: _scalar(value) for key, value in values.items()}
        if step is not None:
            values["progress/update"] = int(step)
        local = {**values, "timestamp": time.time()}
        with (self.directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(local, allow_nan=False) + "\n")
        self.run.log(values)

    def image(self, path, name, step=None):
        if self.rank:
            return
        row = {name: self.wandb.Image(str(path))}
        if step is not None:
            row["progress/update"] = int(step)
        self.run.log(row)

    def artifact(self, path, name, type="report"):
        if self.rank:
            return
        artifact = self.wandb.Artifact(name, type=type)
        path = Path(path)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def finish(self, success=True):
        if self.rank or self.run is None:
            return
        self.metadata["execution_complete"] = bool(success)
        self._write_metadata()
        self.run.summary["execution_complete"] = bool(success)
        self.run.finish(exit_code=0 if success else 1)
        self.run = None
