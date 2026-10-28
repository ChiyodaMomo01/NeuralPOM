# References

This page collects the primary scientific, technical, and software references relevant to the NeuralPOM project. References are organized by topic and presented as a numbered bibliography.

## Princeton Ocean Model (POM)

**[1]** Blumberg, A.F. and Mellor, G.L. (1987). "A description of a three-dimensional coastal ocean circulation model." In: N.S. Heaps (ed.), *Three-Dimensional Coastal Ocean Models*, Coastal and Estuarine Sciences, Vol. 4, pp. 1-16. American Geophysical Union, Washington, D.C. [DOI: 10.1029/CO004p0001](https://doi.org/10.1029/CO004p0001)


**[2]** Mellor, G.L. (2004). "Users guide for a three-dimensional, primitive equation, numerical ocean model." Program in Atmospheric and Oceanic Sciences, Princeton University, Princeton, NJ, 56 pp.


**[3]** Mellor, G.L. (1998). "Users guide for a three-dimensional, primitive equation, numerical ocean model." Program in Atmospheric and Oceanic Sciences, Princeton University, Princeton, NJ, 41 pp.


**[4]** Blumberg, A.F. and Mellor, G.L. (1983). "Diagnostic and prognostic numerical circulation studies of the South Atlantic Bight." *Journal of Geophysical Research: Oceans*, 88(C8): 4579-4592.


## Mellor-Yamada Turbulence Closure

**[5]** Mellor, G.L. and Yamada, T. (1982). "Development of a turbulence closure model for geophysical fluid problems." *Reviews of Geophysics and Space Physics*, 20(4): 851-875. [DOI: 10.1029/RG020i004p00851](https://doi.org/10.1029/RG020i004p00851)


**[6]** Mellor, G.L. and Yamada, T. (1974). "A hierarchy of turbulence closure models for planetary boundary layers." *Journal of the Atmospheric Sciences*, 31(7): 1791-1806.


**[7]** Galperin, B., Kantha, L.H., Hassid, S., and Rosati, A. (1988). "A quasi-equilibrium turbulent energy model for geophysical flows." *Journal of the Atmospheric Sciences*, 45(1): 55-62.


## Arakawa C-Grid and Numerical Methods

**[8]** Arakawa, A. and Lamb, V.R. (1977). "Computational design of the basic dynamical processes of the UCLA general circulation model." In: J. Chang (ed.), *Methods in Computational Physics: Advances in Research and Applications*, Vol. 17, pp. 173-265. Academic Press.


**[9]** Mesinger, F. and Arakawa, A. (1976). "Numerical methods used in atmospheric models." GARP Publication Series No. 17, WMO/ICSU Joint Organizing Committee, 64 pp.


**[10]** Haltiner, G.J. and Williams, R.T. (1980). *Numerical Prediction and Dynamic Meteorology*, 2nd ed. John Wiley & Sons, 477 pp.


**[11]** Asselin, R. (1972). "Frequency filter for time integrations." *Monthly Weather Review*, 100(6): 487-490.


**[12]** Hestenes, M.R. and Stiefel, E. (1952). "Methods of conjugate gradients for solving linear systems." *Journal of Research of the National Bureau of Standards*, 49(6): 409-436.


## AI-Based Weather and Climate Modeling

**[13]** Pathak, J., Subramanian, S., Harrington, P., Raja, S., Chattopadhyay, A., Mardani, M., Kurth, T., Hall, D., Li, Z., Azizzadenesheli, K., Hassanzadeh, P., Kashinath, K., and Anandkumar, A. (2022). "FourCastNet: A global data-driven high-resolution weather model using adaptive Fourier neural operators." arXiv:2202.11214.


**[14]** Lam, R., Sanchez-Gonzalez, A., Willson, M., Wirnsberger, P., Fortunato, M., Alet, F., Ravuri, S., Ewalds, T., Eaton-Rosen, Z., Hu, W., Merose, A., Hoyer, S., Holland, G., Vinyals, O., Stott, J., Pritzel, A., Mohamed, S., and Battaglia, P. (2023). "Learning skillful medium-range global weather forecasting." *Science*, 382(6677): 1416-1421. [DOI: 10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336)


**[15]** Bi, K., Xie, L., Zhang, H., Chen, X., Gu, X., and Tian, Q. (2023). "Accurate medium-range global weather forecasting with 3D neural networks." *Nature*, 619: 533-538. [DOI: 10.1038/s41586-023-06185-3](https://doi.org/10.1038/s41586-023-06185-3)


**[16]** Chen, K., Han, T., Gong, J., Bai, L., Ling, F., Luo, J.-J., Chen, X., Ma, L., Zhang, T., Su, R., Ci, Y., Li, B., Yang, X., and Ouyang, W. (2023). "FengWu: Pushing the skillful global medium-range weather forecast beyond 10 days lead." arXiv:2304.02948.


**[17]** Kochkov, D., Yuval, J., Langmore, I., Norgaard, P., Smith, J., Mooers, G., Klower, M., Lottes, J., Rasp, S., Duben, P., Hatfield, S., Battaglia, P., Sanchez-Gonzalez, A., Willson, M., Brenner, M.P., and Hoyer, S. (2024). "Neural general circulation models for weather and climate." *Nature*, 632: 1060-1066. [DOI: 10.1038/s41586-024-07744-y](https://doi.org/10.1038/s41586-024-07744-y)


## Learned Corrections and Differentiable Physics

**[18]** Um, K., Brand, R., Fei, Y., Holl, P., and Thuerey, N. (2020). "Solver-in-the-Loop: Learning from differentiable physics to interact with iterative PDE-solvers." *Advances in Neural Information Processing Systems*, 33: 6111-6122.


**[19]** List, B., Chen, L.-W., and Thuerey, N. (2022). "Learned turbulence modelling with differentiable fluid solvers: physics-based loss functions and optimisation horizons." *Journal of Fluid Mechanics*, 949: A25.


**[20]** de Bezenac, E., Pajot, A., and Gallinari, P. (2019). "Deep learning for physical processes: incorporating prior scientific knowledge." *Journal of Statistical Mechanics: Theory and Experiment*, 2019(12): 124009.


## Neural Operators for PDEs

**[21]** Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., and Anandkumar, A. (2021). "Fourier neural operator for parametric partial differential equations." *International Conference on Learning Representations* (ICLR).


**[22]** Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., and Anandkumar, A. (2023). "Neural operator: Learning maps between function spaces with applications to PDEs." *Journal of Machine Learning Research*, 24(89): 1-97.


**[23]** Raonic, B., Molinaro, R., De Ryck, T., Rohner, T., Bartolucci, F., Alaifari, R., Mishra, S., and de Bezenac, E. (2024). "Convolutional Neural Operators for robust and accurate learning of PDEs." *Advances in Neural Information Processing Systems*, 36.


## Ocean Modeling and Data Assimilation

**[24]** Griffies, S.M. (2004). *Fundamentals of Ocean Climate Models*. Princeton University Press, 518 pp.


**[25]** Haidvogel, D.B. and Beckmann, A. (1999). *Numerical Ocean Circulation Modeling*. Series on Environmental Science and Management, Vol. 2. Imperial College Press, 344 pp.


**[26]** Shchepetkin, A.F. and McWilliams, J.C. (2005). "The Regional Oceanic Modeling System (ROMS): a split-explicit, free-surface, topography-following-coordinate oceanic model." *Ocean Modelling*, 9(4): 347-404.


## Software

**[27]** Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. (2019). "PyTorch: An imperative style, high-performance deep learning library." *Advances in Neural Information Processing Systems*, 32: 8024-8035.


**[28]** VitePress. (2024). "VitePress: Vite & Vue Powered Static Site Generator." [https://vitepress.dev](https://vitepress.dev)


**[29]** Harris, C.R., Millman, K.J., van der Walt, S.J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N.J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M.H., Brett, M., Haldane, A., del Rio, J.F., Wiebe, M., Peterson, P., Gerard-Marchant, P., Sheppard, K., Reddy, T., Weckesser, W., Abbasi, H., Gohlke, C., and Oliphant, T.E. (2020). "Array programming with NumPy." *Nature*, 585(7825): 357-362. [DOI: 10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)


**[30]** Virtanen, P., Gommers, R., Oliphant, T.E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S.J., Brett, M., Wilson, J., Millman, K.J., Mayorov, N., Nelson, A.R.J., Jones, E., Kern, R., Larson, E., Carey, C.J., Polat, I., Feng, Y., Moore, E.W., VanderPlas, J., Laxalde, D., Perktold, J., Cimrman, R., Henriksen, I., Quintero, E.A., Harris, C.R., Archibald, A.M., Ribeiro, A.H., Pedregosa, F., and van Mulbregt, P. (2020). "SciPy 1.0: Fundamental algorithms for scientific computing in Python." *Nature Methods*, 17(3): 261-272. [DOI: 10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)


**[31]** netCDF4 Development Team. (2024). "netCDF4: Python/numpy interface to the netCDF C library." [https://unidata.github.io/netcdf4-python/](https://unidata.github.io/netcdf4-python/)


## CorrectorNet and NeuralPOM

**[32]** Ronneberger, O., Fischer, P., and Brox, T. (2015). "U-Net: Convolutional networks for biomedical image segmentation." In: *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, Lecture Notes in Computer Science, Vol. 9351, pp. 234-241. Springer.
