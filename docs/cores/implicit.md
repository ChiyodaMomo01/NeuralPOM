# Semi-implicit core

The **semi-implicit core** couples depth-averaged transport and the free surface through an elliptic solve. The production registry name is `implicit_p0_fixed`; its implementation is `src/neuralpom/cores/implicit_p0_fixed/`.

## Implementation

`implicit_barotropic.py` implements the free-surface operator and solve. `momentum.py`, `tracers.py`, `baroclinic.py` and `time_loop.py` supply the remaining POM integration. `autograd.py`, `strict_vertical.py` and `gradient_kernels.py` provide the corresponding differentiation interfaces. See [Elliptic solver](../numerics/elliptic-solver.md).

## Training configuration

The portable templates `configs/examples/implicit_1h.json` and `implicit_1day.json` select the compiled vertical operator and Triton solver backend used by the registered setup. The physical runtime applies the numerical settings through `v1/core_spec.py` and `v1/runtime.py`.

Reference ensembles and calibration assets must correspond to the selected core. The production data reader uses the generation manifest and each member's provenance to identify that pairing; see [Data pipeline](../data/pipeline.md).
