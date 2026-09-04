import json
from pathlib import Path
from givernylocal.turbulence_dataset import turb_dataset
import re

MY_TOKEN = 'your-token-here'

dataset = turb_dataset(
    dataset_title = 'channel5200',
    output_path   = str(Path('data')),
    auth_token    = MY_TOKEN,
)

meta_str = json.dumps(dataset.metadata)
matches = re.findall(r'"code"\s*:\s*"([^"]+)"', meta_str)
print("All codes found in metadata:")
for m in set(matches):
    print(" ", m)