Determined the problem: We start with a high-resolution direct numerical simulation (JHU turbulence database). We want to train a model where given coarser simulation data (numerical simulation with lower resolution), we predict the correction needed to recover fine-grid fidelity.

Procedure:
1. Obtain a 3D cubes of data for training. Apply a Gaussian filter -> downsample -> upsame with spline interpolation, to achieve the same effect as a coarser simulation (since in a coarser simulation, the solver doesn’t “skip” fine-grid points, but rather incorporates a spatially-weighted quantity). Then obtain corresponding labels.
2. CNN model, containing a series of circularly-padded 3D convolutions, normalization, and relu.
3. model(u_coarse_upsampled) produces correction term
4. u_corrected = u_coarse_upsampled + model(u_coarse_upsampled)

Running the code:
1. Obtain the ground truth data from JHU Turbulence DB (run download_data.py).
2. Preprocess the data (generate Gaussian filter and then create model training inputs as well as true label values).
3. Run train.py to train and evaluate model.

