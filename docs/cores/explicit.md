# Explicit core

The **Explicit core** uses the POM separation between the depth-averaged external mode and the three-dimensional internal mode. External free-surface and depth-averaged velocity updates subcycle within each internal step. See [Mode splitting](../physics/barotropic-baroclinic.md) and [Time stepping](../numerics/time-stepping.md) for the formulation.

## Implementation

The production class is `POM2K_Core` in `src/neuralpom/cores/explicit/core.py`. Its modules separate grid construction, initialization, momentum, tracers, baroclinic operators, diagnostics, checkpointing and the integration loop. `core_factory.resolve_core_class("explicit")` returns this class.

## Training configuration

The Explicit templates select eager physical execution with the PyTorch backend. `configs/examples/explicit_1h.json` and `explicit_1day.json` pair the core with hourly and daily endpoint corrections. Reference data retain the Explicit generation settings; [Data format](../data/format.md) describes their storage.
