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


## Build the research guide

The website source is `docs/`. From that directory, run `npm ci` and `npm run docs:build`; `npm run docs:preview` serves the production build locally. The theme components live in `.vitepress/theme/components/`, and the local fonts and manuscript figures in `public/`.

Vercel deploys the repository automatically. Its `cleanUrls` setting must remain enabled so a chapter URL works when opened directly or refreshed. The root and `docs/` Vercel configuration files carry the same routing setting to support either project-root selection. Keep them synchronized.
