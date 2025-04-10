"""
Command Line Interface for EquiForge.
"""

import argparse
import os
import sys
import warnings
import numpy as np
from PIL import Image
from .converters.pers2equi import perspective_to_equirectangular

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
    print(f"Orientation: Pitch={args.pitch}°, Yaw={args.yaw}°, Roll={args.roll}°")
    print(f"Processing mode: {'CPU' if args.cpu else 'GPU if available, otherwise CPU'}")
    
    # Convert image
    equirectangular_image = perspective_to_equirectangular(
        perspective_image,
        args.height,
        fov_h=args.fov,
        pitch=args.pitch,
        yaw=args.yaw,
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
