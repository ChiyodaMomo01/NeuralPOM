# Sources and further reading

This guide links scientific explanations to the **NeuralPOM manuscript** and implementation. The references below support the physical framework and related differentiable-model context. The full manuscript bibliography remains with the paper.

## NeuralPOM

Shu, R., Liu, G., Gou, R., Gao, T., Wen, H., Li, H., and Huang, X. (2026). *New Wine in an Old Bottle — NeuralPOM: A Hybrid Ocean Forecast Model with Differentiable Dynamics and Staged Learning.* arXiv:2611.02460 [physics.ao-ph]. [arXiv](https://arxiv.org/abs/2611.02460), [PDF](https://arxiv.org/pdf/2611.02460), [supplement](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/paper/Supplement.pdf), and [source data](https://github.com/ChiyodaMomo01/NeuralPOM/tree/main/paper/data).

## Ocean dynamics and mixing

**Blumberg, A. F. & Mellor, G. L. (1987).** A description of a three-dimensional coastal ocean circulation model. In *Three-Dimensional Coastal Ocean Models*, pp. 1–16. American Geophysical Union. [doi:10.1029/CO004p0001](https://doi.org/10.1029/CO004p0001). The POM physical and numerical foundation.

**Mellor, G. L. & Yamada, T. (1982).** Development of a turbulence closure model for geophysical fluid problems. *Reviews of Geophysics*, 20, 851–875. [doi:10.1029/RG020i004p00851](https://doi.org/10.1029/RG020i004p00851). The vertical turbulence closure.

**Mellor, G. L. (1991).** An equation of state for numerical models of oceans and estuaries. *Journal of Atmospheric and Oceanic Technology*, 8, 609–611. [Publisher record](https://doi.org/10.1175/1520-0426(1991)008%3C0609:AEOSFN%3E2.0.CO;2). The POM equation of state used to connect tracers and density.

## Differentiable physical models

**Kochkov, D. et al. (2024).** Neural general circulation models for weather and climate. *Nature*, 632, 1060–1066. [doi:10.1038/s41586-024-07744-y](https://doi.org/10.1038/s41586-024-07744-y). A related atmospheric hybrid model; see the NeuralPOM manuscript for the ocean-specific method and comparisons.

## Implementation sources

| Topic | Primary source |
|---|---|
| Configuration and dependencies | [Portable examples](https://github.com/ChiyodaMomo01/NeuralPOM/tree/main/configs/examples), [pyproject.toml](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/pyproject.toml) |
| Physical cores | [Core source tree](https://github.com/ChiyodaMomo01/NeuralPOM/tree/main/src/neuralpom/cores) |
| Training and preparation | [v1 source tree](https://github.com/ChiyodaMomo01/NeuralPOM/tree/main/src/neuralpom/v1) |
| Forecast evaluation | [Evaluation source tree](https://github.com/ChiyodaMomo01/NeuralPOM/tree/main/src/neuralpom/evaluation) |
| Regression checks | [Tests](https://github.com/ChiyodaMomo01/NeuralPOM/tree/main/tests) |
