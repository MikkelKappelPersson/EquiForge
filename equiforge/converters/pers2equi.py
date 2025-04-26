"""
Perspective to Equirectangular Converter

This module converts perspective images to equirectangular projection with optimized performance.
All processing is done using float32 precision for optimal balance of accuracy and performance.
"""

import numpy as np
from PIL import Image
import os
import time
import warnings
import logging
from multiprocessing import Pool, cpu_count
from numba import jit, prange, cuda
from ..utils.projection_utils import create_rotation_matrix, calculate_focal_length, check_cuda_support, timer
from ..utils.logging_utils import setup_logger
from ..utils.sampling import sample_image

# For GPU processing, import GPU sampling methods
HAS_CUDA = check_cuda_support()
if HAS_CUDA:
    from ..utils.sampling import nearest_neighbor_sampling_gpu, bilinear_sampling_gpu

# Set up logger
logger = setup_logger(__name__)

# Define the GPU kernel for perspective to equirectangular conversion
if HAS_CUDA:
    @cuda.jit
    def pers2equi_gpu_kernel(img, equirect, output_width, output_height, 
                         cx, cy, f_h, f_v, w, h, r_matrix, sampling_method):
        """
        CUDA kernel to convert pixels from perspective to equirectangular
        
        Parameters:
        - img: Source perspective image
        - equirect: Output equirectangular image
        - output_width, output_height: Dimensions of output image
        - cx, cy: Center of perspective image
        - f_h, f_v: Focal lengths
        - w, h: Input image dimensions
        - r_matrix: Rotation matrix
        - sampling_method: 0 for nearest neighbor, 1 for bilinear
        """
        x, y = cuda.grid(2)
        
        if x < output_width and y < output_height:
            # Calculate spherical coordinates
            phi = np.pi * y / output_height - np.pi / 2
            theta = 2 * np.pi * x / output_width - np.pi
            
            # CUDA-specific memory allocation
            vec = cuda.local.array(3, dtype=np.float32)
            vec[0] = np.cos(phi) * np.sin(theta)  # x
            vec[1] = np.sin(phi)                  # y
            vec[2] = np.cos(phi) * np.cos(theta)  # z
            
            # CUDA-specific memory handling
            vec_rotated = cuda.local.array(3, dtype=np.float32)
            vec_rotated[0] = r_matrix[0, 0] * vec[0] + r_matrix[0, 1] * vec[1] + r_matrix[0, 2] * vec[2]
            vec_rotated[1] = r_matrix[1, 0] * vec[0] + r_matrix[1, 1] * vec[1] + r_matrix[1, 2] * vec[2]
            vec_rotated[2] = r_matrix[2, 0] * vec[0] + r_matrix[2, 1] * vec[1] + r_matrix[2, 2] * vec[2]
            
            # Only project points in front of the camera (positive z)
            if vec_rotated[2] > 0:
                px = f_h * vec_rotated[0] / vec_rotated[2] + cx
                py = f_v * vec_rotated[1] / vec_rotated[2] + cy
                
                r, g, b = 0, 0, 0
                
                # Use appropriate sampling method based on parameter
                if sampling_method == 1:  # Bilinear
                    # For bilinear we need float coordinates and bounds check with margin
                    if 0 <= px < w-1 and 0 <= py < h-1:
                        r, g, b = bilinear_sampling_gpu(img, py, px, w, h)
                else:  # Nearest neighbor (default)
                    # For nearest we can use direct bounds check
                    if 0 <= px < w and 0 <= py < h:
                        r, g, b = nearest_neighbor_sampling_gpu(img, py, px, w, h)
                
                # Assign sampled values to output
                equirect[y, x, 0] = r
                equirect[y, x, 1] = g
                equirect[y, x, 2] = b

@jit(nopython=True, parallel=True)
def pers2equi_cpu_kernel(img, equirect, output_width, output_height, 
                      x_start, x_end, params, sampling_method="bilinear"):
    """Process a range of columns with Numba optimization on CPU"""
    # Convert to float32 for processing
    img = img.astype(np.float32)
    equirect = equirect.astype(np.float32)
    
    for x in prange(x_start, x_end):
        for y in range(output_height):
            phi = np.pi * y / output_height - np.pi / 2
            theta = 2 * np.pi * x / output_width - np.pi
            
            # Convert spherical to 3D coordinates
            vec_x = np.cos(phi) * np.sin(theta)
            vec_y = np.sin(phi)
            vec_z = np.cos(phi) * np.cos(theta)
            
            # Apply rotation
            r_matrix = params[-1]
            vec_rotated_x = r_matrix[0, 0] * vec_x + r_matrix[0, 1] * vec_y + r_matrix[0, 2] * vec_z
            vec_rotated_y = r_matrix[1, 0] * vec_x + r_matrix[1, 1] * vec_y + r_matrix[1, 2] * vec_z
            vec_rotated_z = r_matrix[2, 0] * vec_x + r_matrix[2, 1] * vec_y + r_matrix[2, 2] * vec_z
            
            # Only project points in front of the camera
            if vec_rotated_z > 0:
                cx, cy, f_h, f_v, w, h = params[:-1]
                px = f_h * vec_rotated_x / vec_rotated_z + cx
                py = f_v * vec_rotated_y / vec_rotated_z + cy
                
                if 0 <= px < w and 0 <= py < h:
                    # Use sampling function
                    equirect[y, x] = sample_image(img, py, px, sampling_method)
    
    # Convert back to uint8 for output
    return np.clip(equirect, 0, 255).astype(np.uint8)

def process_chunk(args):
    """Process a horizontal chunk of the equirectangular image"""
    img, x_start, x_end, output_height, params, sampling_method = args
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    fov_x, yaw, pitch, roll = params[:4]
    
    # Standard equirectangular aspect ratio is 2:1
    output_width = output_height * 2
    
    # Convert angles to radians
    fov_x_rad, yaw_rad, pitch_rad, roll_rad = map(np.radians, [fov_x, yaw, pitch, roll])
    
    # Calculate focal lengths
    f_h, f_v = calculate_focal_length(w, h, fov_x_rad)
    
    # Get rotation matrix
    R = create_rotation_matrix(yaw_rad, pitch_rad, roll_rad)
    
    # Create a chunk of the output image
    chunk = np.zeros((output_height, x_end - x_start, 3), dtype=np.float32)
    
    # Use CPU kernel for processing the chunk
    chunk = pers2equi_cpu_kernel(img, chunk, output_width, output_height, 
                              x_start, x_end, (cx, cy, f_h, f_v, w, h, R), sampling_method)
    
    return chunk, x_start, x_end

def pers2equi_cpu(img, output_height, 
                  fov_x=90.0, yaw=0.0, pitch=0.0, roll=0.0, sampling_method="bilinear"):
    """Multi-threaded conversion from perspective to equirectangular projection"""
    # Standard equirectangular aspect ratio is 2:1
    output_width = output_height * 2
    
    # Validation to ensure image has proper shape
    if len(img.shape) != 3 or img.shape[2] != 3:
        logger.error(f"Input image must have shape (height, width, 3), got {img.shape}")
        raise ValueError(f"Input image must have shape (height, width, 3), got {img.shape}")
    
    # Get optimal number of processes (75% of available CPU cores)
    num_processes = max(1, int(cpu_count() * 0.75))
    
    # Calculate chunk sizes
    chunk_size = max(1, output_width // num_processes)
    num_processes = min(num_processes, output_width) # Adjust if output is smaller than process count
    
    # Prepare arguments for each process
    args_list = []
    for i in range(num_processes):
        x_start = i * chunk_size
        x_end = min(x_start + chunk_size, output_width)
        args_list.append((img, x_start, x_end, output_height, 
                         (fov_x, yaw, pitch, roll), sampling_method))
    
    # Create output equirectangular image
    equirect = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Process chunks in parallel
    logger.info(f"Converting perspective to equirectangular using {num_processes} CPU processes...")
    with Pool(processes=num_processes) as pool:
        results = []
        for chunk_args in args_list:
            results.append(pool.apply_async(process_chunk, (chunk_args,)))
        
        # Monitor progress
        while not all(r.ready() for r in results):
            completed = sum(r.ready() for r in results)
            progress_msg = f"Progress: {completed/len(results)*100:.1f}%"
            logger.debug(progress_msg)
            print(f"{progress_msg}", end="\r")  # Keep real-time console output
            time.sleep(0.5)
        
        # Get results
        for result in results:
            chunk_data, x_start, x_end = result.get()
            
            # Debug output - changed to debug level since empty chunks are expected
            if np.sum(chunk_data) == 0:
                logger.debug(f"Chunk {x_start}-{x_end} contains no image data")
            
            # Copy chunk data to output image
            equirect[:, x_start:x_end] = chunk_data
    
    # Debug output - only warn if the entire output is empty which would be unusual
    if np.sum(equirect) == 0:
        logger.warning("Output image is all zeros - this may indicate a problem with input parameters")
    else:
        logger.info("Conversion completed successfully")
        
    return equirect

@timer
def pers2equi_gpu(img, output_height, 
                 fov_x=90.0, yaw=0.0, pitch=0.0, roll=0.0, sampling_method="bilinear"):
    """GPU-accelerated conversion from perspective to equirectangular projection"""
    # Standard equirectangular aspect ratio is 2:1
    output_width = output_height * 2
    
    logger.info(f"Using GPU acceleration with {sampling_method} sampling...")
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    
    # Convert angles to radians
    fov_x_rad, yaw_rad, pitch_rad, roll_rad = map(np.radians, [fov_x, yaw, pitch, roll])
    
    # Calculate focal lengths
    f_h, f_v = calculate_focal_length(w, h, fov_x_rad)
    
    # Get rotation matrix
    R = create_rotation_matrix(yaw_rad, pitch_rad, roll_rad)
    
    # Create output equirectangular image using float32 for processing
    equirect = np.zeros((output_height, output_width, 3), dtype=np.float32)
    
    # Copy data to GPU - convert inputs to float32 for processing
    logger.debug("Copying data to GPU...")
    d_img = cuda.to_device(img.astype(np.float32))
    d_equirect = cuda.to_device(equirect)
    d_r_matrix = cuda.to_device(R)  # Missing line to copy the rotation matrix to GPU
    
    # Configure grid and block dimensions
    threads_per_block = (16, 16)
    blocks_x = (output_width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (output_height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)
    
    # Convert sampling method string to numeric value
    sampling_method_code = 1 if sampling_method.lower() == "bilinear" else 0
    
    # Launch kernel with appropriate sampling method
    logger.debug(f"Launching CUDA kernel with grid={blocks_per_grid}, block={threads_per_block}")
    pers2equi_gpu_kernel[blocks_per_grid, threads_per_block](
        d_img, d_equirect, output_width, output_height, 
        cx, cy, f_h, f_v, w, h, d_r_matrix, sampling_method_code
    )
    
    # Copy result back to host
    logger.debug("Copying result back to CPU...")
    equirect = d_equirect.copy_to_host()
    
    # Convert back to uint8 for output
    return np.clip(equirect, 0, 255).astype(np.uint8)

def pers2equi(img, output_height, 
              fov_x=90.0, yaw=0.0, pitch=0.0, roll=0.0,
              use_gpu=True, sampling_method="bilinear"):
    """
    Convert perspective image to equirectangular projection
    
    Parameters:
    - img: Input perspective image (numpy array or file path)
    - output_height: Height of output equirectangular image
    - fov_x: Horizontal field of view in degrees
    - yaw: Rotation around vertical axis (left/right) in degrees
    - pitch: Rotation around horizontal axis (up/down) in degrees
    - roll: Rotation around depth axis (clockwise/counterclockwise) in degrees
    - use_gpu: Whether to use GPU acceleration if available
    - sampling_method: Sampling method for pixel interpolation ("nearest" or "bilinear")
    
    Returns:
    - Equirectangular image as numpy array (uint8)
    
    Notes:
    - All internal processing is performed using float32 precision
    - Input images are converted to float32 for processing regardless of input type
    - Output is converted back to uint8 after processing
    - To change logging verbosity, use set_global_log_level() from utils.logging_utils
    """
    # Debug output for better diagnostics
    logger.info(f"pers2equi called with output_height={output_height}, fov_x={fov_x}, use_gpu={use_gpu}")
    
    # Handle file path input
    if isinstance(img, str):
        logger.info(f"Loading image from path: {img}")
        img = np.array(Image.open(img))
        logger.debug(f"Image loaded with shape: {img.shape}")
    
    # Check if img is None
    if img is None:
        raise ValueError("Input image cannot be None")
        
    logger.info(f"Input image shape: {img.shape}, dtype: {img.dtype}")
    
    # Verify input image shape - only accept 3-channel color images
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Input image must have exactly 3 color channels, got shape {img.shape}")
    
    # To ensure computational stability
    fov_x = max(0.1, min(179.9, fov_x))
    logger.info(f"Processing with parameters: FOV={fov_x}°, yaw={yaw}°, pitch={pitch}°, roll={roll}°")
        
    # Determine processing method based on GPU availability and user preference
    result = None
    if use_gpu and HAS_CUDA:
        logger.info("Using GPU processing")
        result = pers2equi_gpu(img, output_height, fov_x, yaw, pitch, roll, sampling_method)
    else:
        if use_gpu and not HAS_CUDA:
            logger.info("GPU requested but not available, using CPU")
        else:
            logger.info("CPU processing requested")
        
        # CPU processing
        logger.info("Starting CPU processing")

        result = pers2equi_cpu(img, output_height, fov_x, yaw, pitch, roll, sampling_method)
    
    logger.info("Processing completed successfully")
    
    return result
