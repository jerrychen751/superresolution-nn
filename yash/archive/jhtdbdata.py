import numpy as np
from pathlib import Path
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getCutout

MY_TOKEN = 'edu.gatech.jerrychen-6b7455a7'

dataset = turb_dataset(
    dataset_title = 'channel5200',
    output_path   = str(Path('data')),
    auth_token    = MY_TOKEN,
)

cutout = getCutout(
    dataset,
    'velocity',
    np.array([
        [0,  31],      # x: 32 points
        [256, 287],    # y: middle of channel, 32 points
        [0,  31],      # z: 32 points
        [1,  1],       # time step 1
    ], dtype=np.int32),
    np.array([1, 1, 1, 1], dtype=np.int32),
)

print("Shape:", cutout.shape)
print("dtype:", cutout.dtype)
print("ux range:", cutout[..., 0].min(), "to", cutout[..., 0].max())
print("uy range:", cutout[..., 1].min(), "to", cutout[..., 1].max())
print("uz range:", cutout[..., 2].min(), "to", cutout[..., 2].max())

# save it
Path('data').mkdir(exist_ok=True)
np.save(Path('data') / 'snapshot_test.npy', cutout.astype(np.float32))
print("Saved to data/snapshot_test.npy")