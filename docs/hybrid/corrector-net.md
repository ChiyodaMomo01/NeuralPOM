# CorrectorNet

## Input and output

**CorrectorNet** is the residual U-Net in `src/neuralpom/models/corrector_net.py`. The hybrid input includes current and historical velocities, temperature and salinity, sea-surface elevation and depth-averaged transports. Six three-dimensional fields over **16 storage levels** and five two-dimensional fields give **101 input channels**.

The output contains increments for zonal velocity, meridional velocity, temperature and salinity over **15 active layers**, giving **60 output channels**. The portable templates use a network width of **256**.

## Physical application

The network output is decoded using the matching training calibration. `hybrid/sparse_coupling.py` applies velocity column constraints, tracer layer budgets and joint limiting, then updates the physical state and density. This corrected endpoint is carried into the following physical interval. The correction map is shared by the [training](./training.md) and frozen forecast interfaces.
