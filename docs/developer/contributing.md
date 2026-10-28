# Contributing

## Development environment

Install the editable package with its test dependencies:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

The regression suite covers core gradients and restart, state/data contracts, sampling, training, frozen evaluation, AI models and tracking. Tests use temporary run directories for generated fixtures.

## Documentation

```bash
cd docs
npm ci
npm run docs:build
```

Keep commands aligned with the current console interfaces and portable templates. Place generated model outputs, checkpoints, local configurations and tracking files in the ignored directories described by `.gitignore`.

## Scientific changes

A change to numerical integration, correction, sampling or checkpoint selection should describe the affected workflow and validation results. Preserve the complete numerical state when changing restart interfaces. Source boundaries are described in [Architecture](./architecture.md).
