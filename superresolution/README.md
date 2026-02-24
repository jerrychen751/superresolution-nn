Determined the problem: We start with a high-resolution direct numerical simulation (JHU turbulence database). We want to train a model where given coarser simulation data (numerical simulation with lower resolution), we predict the correction needed to recover fine-grid fidelity.

Procedure:
1. Obtain a 3D cube of data for training; train with each iteration at a different time step.
2. Apply a Gaussian filter to achieve the same effect as a coarser simulation (since in a coarser simulation, the solver doesn’t “skip” fine-grid points, but rather incorporates a spatially-weighted quantity).
3. Downsample the blurred data. Steps 2 + 3 are data preparation steps to obtain an input for the model from the super-fine DNS to generate synthetic coarse data.
4. model(u_blurred) -> du, dv, dw (correction terms for velocity components)
5. u_corrected = u_blurred + u_model (predict correction terms only for coarse points)

Running the code:
1. Obtain the ground truth data from JHU Turbulence DB (run download_data.py).
2. Preprocess the data (generate Gaussian filter and then create model training inputs as well as true label values).
3. Run train.py to train and evaluate model.

