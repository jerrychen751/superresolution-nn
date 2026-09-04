# import numpy as np
# import torch
# import giverny

# def fetch_jhtdb_channel(token, n_samples=20, size=64):
#     """
#     Queries JHTDB channel5200 using the Giverny library.
    
#     Args:
#         token (str): Your JHTDB/SciServer authentication token.
#         n_samples (int): Number of sub-volumes to fetch.
#         size (int): Resolution of the square slice.
#     """
#     # Set the token for the session
#     giverny.token = token
    
#     data = []
#     dataset = "channel5200"
    
#     print(f"Fetching {n_samples} samples from {dataset} via Giverny...")

#     for i in range(n_samples):
#         # channel5200 Grid: x=10240, y=1536, z=7680
#         # We pick a random X-location and a size x size patch in the Y-Z plane
#         x_idx = np.random.randint(0, 10239)
#         y_start = np.random.randint(0, 1535 - size)
#         z_start = np.random.randint(0, 7679 - size)

#         try:
#             # get_field fetches a specific box. 
#             # Note: Giverny uses [start, end] ranges (inclusive)
#             u_v_w = giverny.get_field(
#                 dataset=dataset,
#                 field="velocity",
#                 time=0,
#                 x_range=[x_idx, x_idx],
#                 y_range=[y_start, y_start + size - 1],
#                 z_range=[z_start, z_start + size - 1]
#             )
            
#             # The result shape is (x, y, z, components) -> (1, 64, 64, 3)
#             # We want (Channels, Height, Width) -> (3, 64, 64)
#             sample = np.squeeze(u_v_w).transpose(2, 0, 1)
#             data.append(sample)
            
#         except Exception as e:
#             print(f"Error at sample {i}: {e}")
#             continue

#     if not data:
#         raise ValueError("No data was fetched. Check your token or network connection.")

#     # Convert to tensor
#     data_tensor = torch.tensor(np.array(data), dtype=torch.float32)
    
#     # Turbulence Normalization:
#     # Since channel5200 velocities vary significantly, we scale to [0, 1]
#     d_min, d_max = data_tensor.min(), data_tensor.max()
#     data_tensor = (data_tensor - d_min) / (d_max - d_min + 1e-6)
    
#     return data_tensor




# import numpy as np
# import torch
# import giverny

# def fetch_jhtdb_channel(token, n_samples=20, size=64):
#     """
#     Queries JHTDB channel5200 using giverny.
#     """
#     giverny.token = token
#     dataset = "channel5200"
#     data = []
    
#     print(f"Targeting JHTDB dataset: {dataset}")

#     for i in range(n_samples):
#         # Coordinates for channel5200
#         x_idx = np.random.randint(0, 10000)
#         y_start = np.random.randint(0, 1536 - size)
#         z_start = np.random.randint(0, 7680 - size)

#         try:
#             # We use the explicit 'get_cutout' function
#             # If this fails, the error message will be very specific
#             u_v_w = giverny.get_cutout(
#                 dataset=dataset,
#                 field="velocity",
#                 time=0,
#                 x_range=[x_idx, x_idx],
#                 y_range=[y_start, y_start + size - 1],
#                 z_range=[z_start, z_start + size - 1]
#             )
            
#             # Convert the downloaded object to a numpy array
#             # Giverny often returns a custom array type; np.array() forces it to standard format
#             sample_np = np.array(u_v_w)
            
#             # Shape is (1, size, size, 3). We want (3, size, size)
#             sample = np.squeeze(sample_np).transpose(2, 0, 1)
#             data.append(sample)
            
#             if (i + 1) % 5 == 0:
#                 print(f"  Successfully fetched {i+1}/{n_samples}")
                
#         except AttributeError:
#             print("CRITICAL ERROR: Giverny was found, but 'get_cutout' is missing.")
#             print(f"Available attributes are: {dir(giverny)}")
#             return None
#         except Exception as e:
#             print(f"Download error at sample {i}: {e}")
#             continue

#     if not data:
#         raise ValueError("Failed to fetch any data. Check your connection/token.")

#     # Convert to Tensor and Normalize
#     data_tensor = torch.tensor(np.array(data), dtype=torch.float32)
#     d_min, d_max = data_tensor.min(), data_tensor.max()
#     return (data_tensor - d_min) / (d_max - d_min + 1e-8)




# import numpy as np
# import torch

# # We attempt to import givernylocal specifically to avoid the "empty" giverny package
# try:
#     import givernylocal as giverny
# except ImportError:
#     import giverny

# def fetch_jhtdb_channel(token, n_samples=20, size=64):
#     """
#     Queries JHTDB channel5200 using the local-access library.
#     """
#     giverny.token = token
#     dataset = "channel5200"
#     data = []
    
#     print(f"Targeting JHTDB dataset: {dataset}")

#     # Verify the library actually has the required function before starting
#     if not hasattr(giverny, 'get_cutout'):
#         print(f"DEBUG: Library attributes found: {dir(giverny)}")
#         raise AttributeError("The 'get_cutout' function is still missing. Ensure givernylocal is installed.")

#     for i in range(n_samples):
#         # Random coordinates within the DNS grid
#         x_idx = np.random.randint(0, 5000) 
#         y_start = np.random.randint(0, 1536 - size)
#         z_start = np.random.randint(0, 5000)

#         try:
#             # get_cutout retrieves a 4D array (x, y, z, components)
#             u_v_w = giverny.get_cutout(
#                 dataset=dataset,
#                 field="velocity",
#                 t_start=0, t_step=1,
#                 x_start=x_idx, x_step=1,
#                 y_start=y_start, y_step=size,
#                 z_start=z_start, z_step=size
#             )
            
#             # The result is (1, size, size, 3). We need (3, size, size)
#             sample_np = np.array(u_v_w)
#             sample = np.squeeze(sample_np).transpose(2, 0, 1)
#             data.append(sample)
            
#             if (i + 1) % 5 == 0:
#                 print(f"  Downloaded {i+1}/{n_samples}...")
                
#         except Exception as e:
#             print(f"Error at sample {i}: {e}")
#             continue

#     if not data:
#         raise ValueError("Fetch failed. No data downloaded.")

#     # Convert and normalize
#     data_tensor = torch.tensor(np.array(data), dtype=torch.float32)
#     d_min, d_max = data_tensor.min(), data_tensor.max()
#     return (data_tensor - d_min) / (d_max - d_min + 1e-8)











import numpy as np
import torch
import os
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getCutout

def fetch_jhtdb_channel(token, n_samples=40, size=64):
    """
    Queries JHTDB channel5200 using the correct givernylocal toolkit.
    """
    # Create a temporary cache directory (local directory) for the library
    cache_dir = "./jhtdb_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # 1. Establish the connection object
    conn = turb_dataset(
        dataset_title="channel5200",
        output_path=cache_dir,
        auth_token=token,
    )
    
    data = []
    print(f"Connection established. Fetching {n_samples} turbulent slices...")

    for i in range(n_samples):
        # channel5200 bounds (1-based): x=10240, y=1536, z=7680
        # We take a random Y-Z plane (64x64) at a random X location
        ox = np.random.randint(1, 10240)
        oy = np.random.randint(1, 1536 - size)
        oz = np.random.randint(1, 7680 - size)
        
        # Time step 1 (Snapshots are usually indexed from 1)
        t_step = 1

        # 2. Define ranges (inclusive, 1-based)
        # axes_ranges shape: [[x_start, x_end], [y_start, y_end], [z_start, z_end], [t_start, t_end]]
        axes_ranges = np.array([
            [ox, ox],               # x (single slice)
            [oy, oy + size - 1],    # y
            [oz, oz + size - 1],    # z
            [t_step, t_step],       # t
        ])
        strides = np.array([1, 1, 1, 1])

        try:
            # 3. Use the toolkit's getCutout function
            result = getCutout(
                conn,
                var="velocity",
                xyzt_axes_ranges_original=axes_ranges,
                xyzt_strides=strides,
            )
            
            # The result is an xarray. Extract the numpy array
            # shape from JHTDB is usually (z, y, x, components)
            var_name = list(result.data_vars)[0]
            velocity = result[var_name].to_numpy() 
            
            # Squeeze to remove the single X and T dimensions
            # Then transpose to (Channels, Height, Width) -> (3, 64, 64)
            sample = np.squeeze(velocity).transpose(2, 0, 1)
            data.append(sample)
            
            if (i + 1) % 5 == 0:
                print(f"  Progress: {i+1}/{n_samples}")
                
        except Exception as e:
            print(f"Error at sample {i}: {e}")
            continue

    if not data:
        raise ValueError("No data fetched. Check your token or database limits.")

    # Convert to tensor and scale to [0, 1]
    data_tensor = torch.tensor(np.array(data), dtype=torch.float32)
    d_min, d_max = data_tensor.min(), data_tensor.max()
    return (data_tensor - d_min) / (d_max - d_min + 1e-8)