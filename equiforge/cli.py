"""
Command Line Interface for EquiForge.
"""

import argparse
import os
import sys
import warnings
import logging
import numpy as np
from PIL import Image
from .converters.pers2equi import pers2equi
from .utils.logging_utils import setup_logger, set_package_log_level

# Setup logger
logger = setup_logger(__name__)

def main():
    """Main entry point for the CLI."""
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
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        default='INFO', help='Set logging level')
    parser.add_argument('--log-file', help='Path to save log file')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level)
    set_package_log_level(log_level)
    
    if args.log_file:
        # Add file handler if log file specified
        file_handler = logging.FileHandler(args.log_file)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {args.log_file}")
    
    logger.info(f"EquiForge CLI started with log level: {args.log_level}")
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file '{args.input}' does not exist")
        return 1
    
    # Load input image
    try:
        logger.info(f"Loading image: {args.input}")
        perspective_image = np.array(Image.open(args.input))
        logger.debug(f"Image loaded with shape: {perspective_image.shape}")
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return 1
    
    # Calculate width based on standard 2:1 aspect ratio
    output_width = args.height * 2
    
    # Display parameters
    logger.info(f"Input image: {args.input} ({perspective_image.shape[1]}x{perspective_image.shape[0]})")
    logger.info(f"Output size: {output_width}x{args.height} (2:1 aspect ratio)")
    logger.info(f"FOV: {args.fov}° (horizontal)")
    logger.info(f"Orientation: Pitch={args.pitch}°, Yaw={args.yaw}°, Roll={args.roll}°")
    logger.info(f"Processing mode: {'CPU' if args.cpu else 'GPU if available, otherwise CPU'}")
    
    # Convert image
    equirectangular_image = pers2equi(
        perspective_image,
        args.height,
        fov_h=args.fov,
        pitch=args.pitch,
        yaw=args.yaw,
        roll=args.roll,
        use_gpu=not args.cpu,
        log_level=log_level
    )
    
    if equirectangular_image is None:
        logger.error("Conversion failed")
        return 1
    
    # Save result
    try:
        logger.info(f"Saving equirectangular image to: {args.output}")
        Image.fromarray(equirectangular_image).save(args.output)
        logger.info("Successfully saved output image")
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return 1
    
    logger.info("Processing completed successfully")
    return 0

if __name__ == "__main__":
    # Suppress warning about invalid values (which can occur but are handled)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sys.exit(main())
