# Cases

The primary training and forecast workflow uses an idealized **wind-driven Double gyre**. The two production dynamical cores share the case settings while retaining their own reference ensembles. See [Double gyre](./double-gyre.md).

The package also retains adapters for coastal tide, Jiaozhou Bay and seamount validation. These are implemented in `src/neuralpom/cores/case_adapters/`, together with their boundary operators and required tidal tables. The manuscript and supplement describe the corresponding numerical comparisons.
