#!/usr/bin/env python3
"""
Perspective to Equirectangular Converter

This script converts a perspective image to equirectangular projection with optimized performance.
"""

import argparse
import numpy as np
from PIL import Image
import os
import sys
import time
from multiprocessing import Pool, cpu_count
import warnings
from numba import jit, prange, cuda

# Check for CUDA support
HAS_CUDA = cuda.is_available()
if HAS_CUDA:
    print("CUDA support found: GPU acceleration available")
else:
    print("No compatible GPU detected, using CPU acceleration only")

# Define CUDA kernel for GPU acceleration
if HAS_CUDA:
    @cuda.jit
    def equirect_gpu_kernel(img, equirect, output_width, output_height, 
                         cx, cy, f_h, f_v, w, h, r_matrix):
        """CUDA kernel to convert pixels from perspective to equirectangular"""
        x, y = cuda.grid(2)
        
        if x < output_width and y < output_height:
            # Calculate spherical coordinates
            phi = np.pi * y / output_height - np.pi / 2
            theta = 2 * np.pi * x / output_width - np.pi
            
            # CUDA-specific memory allocation - can't be abstracted out
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
                px = int(f_h * vec_rotated[0] / vec_rotated[2] + cx)
                py = int(f_v * vec_rotated[1] / vec_rotated[2] + cy)
                
                # Check if within perspective image bounds
                if 0 <= px < w and 0 <= py < h:
                    # Copy pixel values
                    equirect[y, x, 0] = img[py, px, 0]
                    equirect[y, x, 1] = img[py, px, 1]
                    equirect[y, x, 2] = img[py, px, 2]

@jit(nopython=True, parallel=True)
def equirect_cpu_kernel(img, equirect, output_width, output_height, 
                      y_start, y_end, cx, cy, f_h, f_v, w, h, r_matrix):
    """Process a range of rows with Numba optimization on CPU"""
    for y in prange(y_start, y_end):
        phi = np.pi * y / output_height - np.pi / 2
        
        for x in range(output_width):
            theta = 2 * np.pi * x / output_width - np.pi
            
            # Convert spherical to 3D coordinates
            vec_x = np.cos(phi) * np.sin(theta)
            vec_y = np.sin(phi)
            vec_z = np.cos(phi) * np.cos(theta)
            
            # Apply rotation
            vec_rotated_x = r_matrix[0, 0] * vec_x + r_matrix[0, 1] * vec_y + r_matrix[0, 2] * vec_z
            vec_rotated_y = r_matrix[1, 0] * vec_x + r_matrix[1, 1] * vec_y + r_matrix[1, 2] * vec_z
            vec_rotated_z = r_matrix[2, 0] * vec_x + r_matrix[2, 1] * vec_y + r_matrix[2, 2] * vec_z
            
            # Only project points in front of the camera
            if vec_rotated_z > 0:
                px = int(f_h * vec_rotated_x / vec_rotated_z + cx)
                py = int(f_v * vec_rotated_y / vec_rotated_z + cy)
                
                if 0 <= px < w and 0 <= py < h:
                    # Copy pixel values
                    equirect[y, x, 0] = img[py, px, 0]
                    equirect[y, x, 1] = img[py, px, 1]
                    equirect[y, x, 2] = img[py, px, 2]
    
    return equirect

def process_chunk(args):
    """Process a horizontal chunk of the equirectangular image"""
    img, y_start, y_end, output_height, params = args
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    fov_h, yaw, pitch, roll = params
    
    # Standard equirectangular aspect ratio is 2:1
    output_width = output_height * 2
    
    # Calculate vertical FOV based on aspect ratio
    aspect_ratio = w / h
    fov_v = fov_h / aspect_ratio
    
    # Convert angles to radians
    fov_h_rad, fov_v_rad = np.radians(fov_h), np.radians(fov_v)
    yaw_rad, pitch_rad, roll_rad = np.radians(yaw), np.radians(pitch), np.radians(roll)
    
    # Calculate focal lengths
    f_h = (w / 2) / np.tan(fov_h_rad / 2)
    f_v = (h / 2) / np.tan(fov_v_rad / 2)
    
    # Create rotation matrices
    R_yaw = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    R_roll = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = R_yaw @ R_pitch @ R_roll
    
    # Create a chunk of the output image
    chunk = np.zeros((y_end - y_start, output_width, 3), dtype=np.uint8)
    
    # Use CPU kernel for processing the chunk
    chunk = equirect_cpu_kernel(img, chunk, output_width, output_height, 
                              0, y_end - y_start, cx, cy, f_h, f_v, w, h, R)
    
    return chunk


def perspective_to_equirectangular_parallel(img, output_height, 
                                          fov_h=90.0, yaw=0.0, pitch=0.0, roll=0.0):
    """
    Multi-threaded conversion from perspective to equirectangular projection
    """
    # Standard equirectangular aspect ratio is 2:1
    output_width = output_height * 2
    
    # Validation to ensure image has proper shape
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Input image must have shape (height, width, 3), got {img.shape}")
    
    # Get optimal number of processes (75% of available CPU cores)
    num_processes = max(1, int(cpu_count() * 0.75))
    
    # Calculate chunk sizes
    chunk_size = max(1, output_height // num_processes)
    num_processes = min(num_processes, output_height) # Adjust if output is smaller than process count
    
    # Prepare arguments for each process
    args_list = []
    for i in range(num_processes):
        y_start = i * chunk_size
        y_end = min(y_start + chunk_size, output_height)
        args_list.append((img, y_start, y_end, output_height, (fov_h, yaw, pitch, roll)))
    
    # Create output equirectangular image
    equirect = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Process chunks in parallel
    print(f"Converting perspective to equirectangular using {num_processes} CPU processes...")
    with Pool(processes=num_processes) as pool:
        results = []
        for i, chunk_args in enumerate(args_list):
            results.append(pool.apply_async(process_chunk, (chunk_args,)))
        
        # Monitor progress
        while not all(r.ready() for r in results):
            completed = sum(r.ready() for r in results)
            print(f"Progress: {completed/len(results)*100:.1f}%", end="\r")
            time.sleep(0.5)
        
        # Get results
        for i, result in enumerate(results):
            y_start = i * chunk_size
            y_end = min(y_start + chunk_size, output_height)
            chunk_data = result.get()
            
            # Debug output
            if np.sum(chunk_data) == 0:
                print(f"Warning: Chunk {i} is all zeros")
            
            # Copy chunk data to output image
            equirect[y_start:y_end] = chunk_data
    
    # Debug output
    if np.sum(equirect) == 0:
        print("Warning: Output image is all zeros!")
    else:
        print("Conversion complete with non-zero data!")
        
    return equirect


def perspective_to_equirectangular_gpu(img, output_height, 
                                    fov_h=90.0, yaw=0.0, pitch=0.0, roll=0.0):
    """
    Convert perspective image to equirectangular projection using GPU acceleration
    
    Parameters:
    - img: Input perspective image (numpy array)
    - output_height: Height of output equirectangular image
    - fov_h: Horizontal field of view in degrees
    - yaw: Rotation around vertical axis (left/right) in degrees
    - pitch: Rotation around horizontal axis (up/down) in degrees
    - roll: Rotation around depth axis (clockwise/counterclockwise) in degrees
    
    Returns:
    - Equirectangular image as numpy array
    """
    # Standard equirectangular aspect ratio is 2:1
    output_width = output_height * 2
    
    print("Using GPU acceleration...")
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    
    # Calculate vertical FOV based on aspect ratio
    aspect_ratio = w / h
    fov_v = fov_h / aspect_ratio
    
    # Convert angles to radians
    fov_h_rad, fov_v_rad = np.radians(fov_h), np.radians(fov_v)
    yaw_rad, pitch_rad, roll_rad = np.radians(yaw), np.radians(pitch), np.radians(roll)
    
    # Calculate focal lengths
    f_h = (w / 2) / np.tan(fov_h_rad / 2)
    f_v = (h / 2) / np.tan(fov_v_rad / 2)
    
    # Create rotation matrices
    R_yaw = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    R_roll = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = R_yaw @ R_pitch @ R_roll
    
    # Create output equirectangular image
    equirect = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Copy data to GPU
    d_img = cuda.to_device(img)
    d_equirect = cuda.to_device(equirect)
    d_r_matrix = cuda.to_device(R)
    
    # Configure grid and block dimensions
    threads_per_block = (16, 16)
    blocks_x = (output_width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (output_height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)
    
    # Launch kernel
    equirect_gpu_kernel[blocks_per_grid, threads_per_block](
        d_img, d_equirect, output_width, output_height, 
        cx, cy, f_h, f_v, w, h, d_r_matrix
    )
    
    # Copy result back to host
    equirect = d_equirect.copy_to_host()
    
    return equirect

def perspective_to_equirectangular(img, output_height, 
                                  fov_h=90.0, yaw=0.0, pitch=0.0, roll=0.0,
                                  use_gpu=True):
    """
    Convert perspective image to equirectangular projection
    
    Parameters:
    - img: Input perspective image (numpy array or file path)
    - output_height: Height of output equirectangular image
    - fov_h: Horizontal field of view in degrees
    - yaw: Rotation around vertical axis (left/right) in degrees
    - pitch: Rotation around horizontal axis (up/down) in degrees
    - roll: Rotation around depth axis (clockwise/counterclockwise) in degrees
    - use_gpu: Whether to use GPU acceleration if available
    
    Returns:
    - Equirectangular image as numpy array
    """
    start_time = time.time()
    
    # Handle file path input
    if isinstance(img, str):
        try:
            print(f"Loading image from path: {img}")
            img = np.array(Image.open(img))
        except Exception as e:
            print(f"Error loading image from path: {e}")
            return None
    
    # Standard equirectangular aspect ratio is 2:1
    output_width = output_height * 2
    
    # Verify input image shape
    if len(img.shape) != 3 or img.shape[2] != 3:
        print(f"Warning: Expected 3-channel color image, got shape {img.shape}")
        # Try to fix if grayscale
        if len(img.shape) == 2:
            img = np.stack((img,)*3, axis=-1)
    
    # To ensure computational stability
    fov_h = max(0.1, min(179.9, fov_h))
    
    try:
        if use_gpu and HAS_CUDA:
            try:
                result = perspective_to_equirectangular_gpu(img, output_height, 
                                                            fov_h, yaw, pitch, roll)
                # Verify we have actual data
                if np.sum(result) == 0:
                    raise ValueError("GPU processing returned an all-zero image. Falling back to CPU.")
                print(f"GPU processing completed in {time.time() - start_time:.2f} seconds.")
                return result
            except Exception as e:
                print(f"GPU processing failed: {e}. Falling back to CPU.")
        
        # Fallback to CPU processing
        result = perspective_to_equirectangular_parallel(img, output_height, 
                                                         fov_h, yaw, pitch, roll)
        print(f"CPU processing completed in {time.time() - start_time:.2f} seconds.")
        return result
    except Exception as e:
        print(f"Error during processing: {e}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert perspective image to equirectangular projection')
    parser.add_argument('input', help='Input perspective image path')
    parser.add_argument('output', help='Output equirectangular image path')
    parser.add_argument('--fov', type=float, default=90.0, help='Horizontal field of view in degrees (default: 90.0)')
    parser.add_argument('--height', type=int, default=2048, help='Output height (default: 2048)')
    parser.add_argument('--yaw', type=float, default=0.0, help='Yaw rotation in degrees (default: 0.0)')
    parser.add_argument('--pitch', type=float, default=0.0, help='Pitch rotation in degrees (default: 0.0)')
    parser.add_argument('--roll', type=float, default=0.0, help='Roll rotation in degrees (default: 0.0)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU processing instead of GPU')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return 1
    
    # Load input image
    try:
        print(f"Loading image: {args.input}")
        perspective_image = np.array(Image.open(args.input))
    except Exception as e:
        print(f"Error loading image: {e}")
        return 1
    
    # Calculate width based on standard 2:1 aspect ratio
    output_width = args.height * 2
    
    # Display parameters
    print(f"Input image: {args.input} ({perspective_image.shape[1]}x{perspective_image.shape[0]})")
    print(f"Output size: {output_width}x{args.height} (2:1 aspect ratio)")
    print(f"FOV: {args.fov}° (horizontal)")
    print(f"Orientation: Yaw={args.yaw}°, Pitch={args.pitch}°, Roll={args.roll}°")
    print(f"Processing mode: {'CPU' if args.cpu else 'GPU if available, otherwise CPU'}")
    
    # Convert image
    equirectangular_image = perspective_to_equirectangular(
        perspective_image,
        args.height,
        fov_h=args.fov,
        yaw=args.yaw,
        pitch=args.pitch,
        roll=args.roll,
        use_gpu=not args.cpu
    )
    
    # Save result
    try:
        print(f"Saving equirectangular image to: {args.output}")
        Image.fromarray(equirectangular_image).save(args.output)
        print("Done!")
    except Exception as e:
        print(f"Error saving image: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Suppress warning about invalid values (which can occur but are handled)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sys.exit(main())
