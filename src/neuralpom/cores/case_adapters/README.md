# Physical case adapters

The adapters configure the Explicit and Implicit POM frameworks for four reference cases. Each adapter installs the case's boundary conditions and numerical settings on a model instance.

| Case | Stored grid | External / internal time step |
|---|---|---|
| `double_gyre` | 90 × 82 × 16 | 30 / 900 s; additional 10 / 900, 10 / 300 and 10 / 100 s configurations |
| `coastal_tide` | 161 × 101 × 5 | 1 / 10 s |
| `seamount` | 65 × 49 × 16 | 5 / 150 s |
| `jiaozhou_bay` | 312 × 213 × 17 | 1 / 10 s |

## Reference-state integration

Supply a complete reference initial state, including the current and previous time levels:

```python
import torch
from neuralpom.cores.case_adapters import core_from_reference

core = core_from_reference(
    "initial_state.npz", case="coastal_tide", family="implicit"
)
core.iend = 1
with torch.no_grad():
    core.run_time_loop()
```

`apply_case_adaptation` configures an already initialized instance. The reference interfaces use double-precision states and preserve the case's geometry, clocks and boundary forcing. The tidal cases load the harmonic tables packaged in `data/`; `tide_path` selects an alternative table when `strict=False`.

The double-gyre adapter sets the closed-basin transport and mixing operators. Coastal-tide and Jiaozhou Bay adapters apply prescribed tidal elevation, radiation boundaries and bottom friction. The seamount adapter applies open-boundary and free-surface settings for the stratified topographic case.

`source_manifest.json` records the original and exported implementation hashes and package import mappings. The physical comparisons are described in Appendix F of the [paper](../../../../paper/NeuralPOM.pdf).
