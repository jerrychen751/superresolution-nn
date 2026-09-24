# CNN preprocessing: DNS velocity to training pairs

Preprocessing turns each high-resolution JHTDB velocity cube into an
**input–answer pair** for the residual CNN.

Each raw cube has shape:

```text
128 × 128 × 128 × 3
```

The final dimension contains the three velocity components \(u,v,w\).

For each cube, preprocessing performs this pipeline:

```text
Original DNS velocity: 128³
          │
          ▼
Gaussian smoothing, σ = 1
          │
          ▼
Keep every 4th point: 32³
          │
          ▼
Cubic-spline upsample back to 128³
          │
          ├── CNN input
          │
          ▼
Original DNS − CNN input
          │
          └── CNN target correction
```

Mathematically,

\[
u_{\text{input}}
=
\operatorname{Upsample}
\left(
\operatorname{Downsample}
\left(
\operatorname{Gaussian}(u_{\text{DNS}})
\right)
\right)
\]

\[
u_{\text{correction}}
=
u_{\text{DNS}}-u_{\text{input}}
\]

The CNN learns:

\[
\operatorname{CNN}(u_{\text{input}})
\approx
u_{\text{correction}}
\]

The reconstructed fine velocity is:

\[
u_{\text{reconstructed}}
=
u_{\text{input}}+\operatorname{CNN}(u_{\text{input}})
\]

Gaussian smoothing happens before downsampling to reduce aliasing. Without it,
simply keeping every fourth point could turn unresolved small structures into
false coarse-scale patterns.

The implementation applies the Gaussian filter independently to all three
velocity components with periodic wrapping. It then samples every fourth point
along all three spatial axes and uses periodic cubic-spline interpolation to
return to the original grid size.

With the current 700-cube team dataset, preprocessing sorts cubes by timestep
and creates a chronological split:

```text
train: 490 pairs
validation: 105 pairs
test: 105 pairs
```

The saved arrays are located under `data/processed/cnn/`. Each `input_t*.npy`
contains \(u_{\text{input}}\), and its matching `target_t*.npy` contains
\(u_{\text{correction}}\).

Preprocessing does not train the neural network, normalize velocities, add
noise, randomly crop cubes, or augment the data. Training begins later in the
CNN notebook.
