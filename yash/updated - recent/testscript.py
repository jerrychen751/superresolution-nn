import givernylocal
import numpy as np

# This should now show many more attributes (get_cutout, get_raw_data, etc.)
print("Available attributes:", dir(givernylocal))

if hasattr(givernylocal, 'get_cutout'):
    print("SUCCESS: 'get_cutout' is now available.")
    
    # Optional: Quick test fetch (Channel5200)
    # This fetches a small 8x8x8 cube of velocity (u,v,w)
    try:
        data = givernylocal.get_cutout(
            dataset='channel5200',
            field='u',
            t_start=0, t_step=1,
            x_start=0, x_step=8,
            y_start=0, y_step=8,
            z_start=0, z_step=8
        )
        print(f"Data fetched successfully! Shape: {data.shape}")
    except Exception as e:
        print(f"Fetch failed, but the library is working. Error: {e}")
else:
    print("CRITICAL: 'get_cutout' still missing.")