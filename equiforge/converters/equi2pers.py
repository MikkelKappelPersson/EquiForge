"""
Equirectangular to Perspective Converter

This module converts equirectangular projections to perspective images with optimized performance.
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

# Set up logger
logger = setup_logger(__name__)

# Check for CUDA support
HAS_CUDA = check_cuda_support()

# Update imports to include GPU sampling methods
if HAS_CUDA:
    from ..utils.sampling import nearest_neighbor_sampling_gpu, bilinear_sampling_gpu

# Define CUDA kernel for GPU acceleration
if HAS_CUDA:
    @cuda.jit
    def equi2pers_gpu_kernel(equi, perspective, output_width, output_height, 
                         cx, cy, f_h, f_v, equi_w, equi_h, r_matrix, sampling_method):
        """CUDA kernel to convert pixels from equirectangular to perspective"""
        x, y = cuda.grid(2)
        
        if x < output_width and y < output_height:
            # Calculate normalized device coordinates
            # Map x from [0, output_width] to [-1, 1]
            # Map y from [0, output_height] to [-1, 1] 
            # Note: No need to flip y-axis as we're directly using the correct coordinate system
            ndc_x = (x - cx) / f_h
            ndc_y = (y - cy) / f_v  # Not flipped - this was causing the upside-down issue
            
            # Create direction vector in camera space
            dir_x = ndc_x
            dir_y = -ndc_y  # Negate y to match OpenGL/standard 3D graphics convention
            dir_z = 1.0  # Forward direction
            
            # Normalize the direction vector
            length = cuda.libdevice.sqrtf(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)
            dir_x /= length
            dir_y /= length
            dir_z /= length
            
            # Apply inverse rotation to get world direction
            # CUDA-specific memory handling
            world_dir = cuda.local.array(3, dtype=np.float32)
            world_dir[0] = r_matrix[0, 0] * dir_x + r_matrix[1, 0] * dir_y + r_matrix[2, 0] * dir_z
            world_dir[1] = r_matrix[0, 1] * dir_x + r_matrix[1, 1] * dir_y + r_matrix[2, 1] * dir_z
            world_dir[2] = r_matrix[0, 2] * dir_x + r_matrix[1, 2] * dir_y + r_matrix[2, 2] * dir_z
            
            # Convert to spherical coordinates
            theta = np.arctan2(world_dir[0], world_dir[2])  # longitude (yaw)
            phi = np.arcsin(world_dir[1])                  # latitude (pitch)
            
            # Map spherical coordinates to equirectangular pixel coordinates
            # Convert theta from [-pi, pi] to [0, equi_w]
            equi_x = ((theta + np.pi) / (2 * np.pi) * equi_w) % equi_w
            
            # Convert phi from [-pi/2, pi/2] to [0, equi_h]
            equi_y = ((np.pi/2 - phi) / np.pi * equi_h) % equi_h
            
            r, g, b = 0, 0, 0
            
            # Use appropriate sampling method based on parameter
            if sampling_method == 1:  # Bilinear
                # For bilinear we need float coordinates and bounds check with margin
                if 0 <= equi_x < equi_w-1 and 0 <= equi_y < equi_h-1:
                    r, g, b = bilinear_sampling_gpu(equi, equi_y, equi_x, equi_w, equi_h)
            else:  # Nearest neighbor (default)
                # For nearest we can use direct bounds check
                if 0 <= equi_x < equi_w and 0 <= equi_y < equi_h:
                    r, g, b = nearest_neighbor_sampling_gpu(equi, equi_y, equi_x, equi_w, equi_h)
            
            # Assign sampled values to output
            perspective[y, x, 0] = r
            perspective[y, x, 1] = g
            perspective[y, x, 2] = b

@jit(nopython=True, parallel=True)
def equi2pers_cpu_kernel(equi, perspective, output_width, output_height, 
                       x_start, x_end, cx, cy, f_h, f_v, equi_w, equi_h, r_matrix_inv, sampling_method="bilinear"):
    """Process a range of columns with Numba optimization on CPU"""
    # Convert to float32 for processing
    perspective = perspective.astype(np.float32)
    equi = equi.astype(np.float32)
    
    chunk_width = x_end - x_start
    for chunk_x in prange(chunk_width):
        x = x_start + chunk_x  # Convert local chunk coordinate to global image coordinate
        
        for y in range(output_height):
            # Calculate normalized device coordinates with standard y-axis orientation
            ndc_x = (x - cx) / f_h
            ndc_y = (y - cy) / f_v  # Not flipped - use standard orientation
            
            # Create direction vector in camera space
            dir_x = ndc_x
            dir_y = -ndc_y  # Negate y to match standard 3D graphics convention
            dir_z = 1.0  # Forward direction
            
            # Normalize the direction vector
            length = np.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)
            dir_x /= length
            dir_y /= length
            dir_z /= length
            
            # Apply inverse rotation to get world direction
            world_dir_x = r_matrix_inv[0, 0] * dir_x + r_matrix_inv[0, 1] * dir_y + r_matrix_inv[0, 2] * dir_z
            world_dir_y = r_matrix_inv[1, 0] * dir_x + r_matrix_inv[1, 1] * dir_y + r_matrix_inv[1, 2] * dir_z
            world_dir_z = r_matrix_inv[2, 0] * dir_x + r_matrix_inv[2, 1] * dir_y + r_matrix_inv[2, 2] * dir_z
            
            # Convert to spherical coordinates
            theta = np.arctan2(world_dir_x, world_dir_z)  # longitude (yaw)
            phi = np.arcsin(world_dir_y)                  # latitude (pitch)
            
            # Map spherical coordinates to equirectangular pixel coordinates with standard orientation
            equi_x = int((theta + np.pi) / (2 * np.pi) * equi_w) % equi_w
            equi_y = int((np.pi/2 - phi) / np.pi * equi_h) % equi_h  # Standard mapping for equirectangular
            
            # Use sampling function for pixel assignment - write to local chunk coordinate
            perspective[y, chunk_x] = sample_image(equi, equi_y, equi_x, sampling_method)
    
    # Convert back to uint8 for output
    return np.clip(perspective, 0, 255).astype(np.uint8)

def process_chunk(args):
    """Process a horizontal chunk of the perspective image"""
    equi, x_start, x_end, output_width, output_height, params, sampling_method = args
    equi_h, equi_w = equi.shape[:2]
    fov_x, yaw, pitch, roll = params
    
    # Calculate center of output image
    cx = output_width // 2
    cy = output_height // 2
    
    # Convert angles to radians
    fov_x_rad, yaw_rad, pitch_rad, roll_rad = map(np.radians, [fov_x, yaw, pitch, roll])
    
    # Calculate focal lengths
    f_h, f_v = calculate_focal_length(output_width, output_height, fov_x_rad)
    
    # Get rotation matrix and its inverse
    R = create_rotation_matrix(yaw_rad, pitch_rad, roll_rad)
    R_inv = np.linalg.inv(R)
    
    # Create a chunk of the output image
    chunk_width = x_end - x_start
    chunk = np.zeros((output_height, chunk_width, 3), dtype=np.float32)
    
    # Use CPU kernel for processing the chunk - fixed to use proper coordinates
    chunk = equi2pers_cpu_kernel(equi, chunk, output_width, output_height, 
                              x_start, x_end, cx, cy, f_h, f_v, equi_w, equi_h, R_inv, sampling_method)
    
    return chunk, x_start, x_end

def equi2pers_cpu(equi, output_width, output_height,
                  fov_x=90.0, yaw=0.0, pitch=0.0, roll=0.0, 
                  sampling_method="bilinear"):
    """Multi-threaded conversion from equirectangular to perspective projection"""
    # Validation to ensure image has proper shape
    if len(equi.shape) != 3 or equi.shape[2] != 3:
        logger.error(f"Input image must have shape (height, width, 3), got {equi.shape}")
        raise ValueError(f"Input image must have shape (height, width, 3), got {equi.shape}")
    
    # Get optimal number of processes (75% of available CPU cores)
    num_processes = max(1, int(cpu_count() * 0.75))
    
    # Calculate chunk sizes - divide the width into chunks
    chunk_size = max(1, output_width // num_processes)
    num_processes = min(num_processes, output_width) # Adjust if output is smaller than process count
    
    # Prepare arguments for each process
    args_list = []
    for i in range(num_processes):
        x_start = i * chunk_size
        x_end = min(x_start + chunk_size, output_width)
        args_list.append((equi, x_start, x_end, output_width, output_height, 
                         (fov_x, yaw, pitch, roll), sampling_method))
    
    # Create output perspective image
    perspective = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Process chunks in parallel
    logger.info(f"Converting equirectangular to perspective using {num_processes} CPU processes...")
    with Pool(processes=num_processes) as pool:
        results = []
        for i, chunk_args in enumerate(args_list):
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
            
            # Copy chunk data to output image
            perspective[:, x_start:x_end] = chunk_data
    
    logger.info("CPU conversion completed successfully")
    return perspective

@timer
def equi2pers_gpu(equi, output_width, output_height,
                 fov_x=90.0, yaw=0.0, pitch=0.0, roll=0.0, sampling_method="bilinear"):
    """GPU-accelerated conversion from equirectangular to perspective projection"""
    logger.info("Using GPU acceleration...")
    equi_h, equi_w = equi.shape[:2]
    
    # Calculate center of output image
    cx = output_width // 2
    cy = output_height // 2
    
    # Convert angles to radians
    fov_x_rad, yaw_rad, pitch_rad, roll_rad = map(np.radians, [fov_x, yaw, pitch, roll])
    
    # Calculate focal lengths
    f_h, f_v = calculate_focal_length(output_width, output_height, fov_x_rad)
    
    # Get rotation matrix
    R = create_rotation_matrix(yaw_rad, pitch_rad, roll_rad)
    R_inv = np.linalg.inv(R)
    
    # Create output perspective image (using float32 for processing)
    perspective = np.zeros((output_height, output_width, 3), dtype=np.float32)
    
    # Copy data to GPU - convert inputs to float32 for processing
    logger.debug("Copying data to GPU...")
    d_equi = cuda.to_device(equi.astype(np.float32))
    d_perspective = cuda.to_device(perspective)
    d_r_matrix = cuda.to_device(R_inv)  # Use the inverse rotation matrix
    
    # Configure grid and block dimensions
    threads_per_block = (16, 16)
    blocks_x = (output_width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (output_height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)
    
    # Convert sampling method string to numeric value
    sampling_method_code = 1 if sampling_method.lower() == "bilinear" else 0
    
    # Launch kernel with sampling method parameter
    logger.debug(f"Launching CUDA kernel with grid={blocks_per_grid}, block={threads_per_block}")
    equi2pers_gpu_kernel[blocks_per_grid, threads_per_block](
        d_equi, d_perspective, output_width, output_height, 
        cx, cy, f_h, f_v, equi_w, equi_h, d_r_matrix, sampling_method_code
    )
    
    # Copy result back to host and convert to uint8
    logger.debug("Copying result back to CPU...")
    perspective = d_perspective.copy_to_host()
    
    # Convert back to uint8 for output
    return np.clip(perspective, 0, 255).astype(np.uint8)

def equi2pers(img, output_width, output_height,
              fov_x=90.0, yaw=0.0, pitch=0.0, roll=0.0,
              use_gpu=True, sampling_method="bilinear"):
    """
    Convert equirectangular image to perspective projection
    
    Parameters:
    - img: Input equirectangular image (numpy array or file path)
    - output_width: Width of output perspective image
    - output_height: Height of output perspective image
    - fov_x: Horizontal field of view in degrees
    - yaw: Rotation around vertical axis (left/right) in degrees
    - pitch: Rotation around horizontal axis (up/down) in degrees
    - roll: Rotation around depth axis (clockwise/counterclockwise) in degrees
    - use_gpu: Whether to use GPU acceleration if available
    - sampling_method: Sampling method for pixel interpolation (default: "bilinear")
    
    Returns:
    - Perspective image as numpy array (uint8)
    
    Notes:
    - All internal processing is performed using float32 precision
    - Input images are converted to float32 for processing regardless of input type
    - Output is converted back to uint8 after processing
    - To change logging verbosity, use set_global_log_level() from utils.logging_utils
    """
    # Debug output for better diagnostics
    logger.info(f"equi2pers called with output_width={output_width}, output_height={output_height}, fov_x={fov_x}, use_gpu={use_gpu}")
    
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
        result = equi2pers_gpu(img, output_width, output_height, fov_x, yaw, pitch, roll, sampling_method)
    else:
        if use_gpu and not HAS_CUDA:
            logger.info("GPU requested but not available, using CPU")
        else:
            logger.info("CPU processing requested")
        
        # CPU processing
        logger.info("Starting CPU processing")
        result = equi2pers_cpu(img, output_width, output_height, fov_x, yaw, pitch, roll, sampling_method)
    
    logger.info("Processing completed successfully")
    
    return result
