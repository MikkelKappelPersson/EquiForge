"""
Visualization tool for test images used in EquiForge

This script generates and displays the test images used in EquiForge's test suite,
allowing developers to visually understand the test data.
"""

import numpy as np
import matplotlib.pyplot as plt

def create_perspective_test_image():
    """Create the same test perspective image used in tests"""
    # Create a 100x100 test image with a gradient pattern
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            img[i, j] = [i, j, (i+j)//2]
    return img

def create_equirectangular_test_image():
    """Create the same test equirectangular image used in tests"""
    # Create a 200x100 equirectangular test image (2:1 ratio)
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(200):
            img[i, j] = [i, j % 100, (i+j)//2]
    return img

if __name__ == "__main__":
    # Create both test images
    pers_img = create_perspective_test_image()
    equi_img = create_equirectangular_test_image()
    
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display perspective test image
    ax1.imshow(pers_img)
    ax1.set_title('Perspective Test Image (100x100)')
    ax1.set_xlabel('j coordinate (green channel)')
    ax1.set_ylabel('i coordinate (red channel)')
    
    # Display equirectangular test image
    ax2.imshow(equi_img)
    ax2.set_title('Equirectangular Test Image (200x100)')
    ax2.set_xlabel('j coordinate (green channel cycles at 100)')
    ax2.set_ylabel('i coordinate (red channel)')
    
    # Add an overall title
    fig.suptitle('EquiForge Test Images', fontsize=16)
    
    # Add description text explaining the color mapping
    text = """
    Color mapping:
    - Red channel (R) increases vertically (i coordinate)
    - Green channel (G) increases horizontally (j coordinate)
    - Blue channel (B) is the average of R and G divided by 2
    """
    fig.text(0.5, 0.01, text, ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
    
    print("The perspective test image has these pixel values at key locations:")
    print(f"Top-left (0,0): {pers_img[0, 0]} (Black)")
    print(f"Top-right (0,99): {pers_img[0, 99]} (Green)")
    print(f"Bottom-left (99,0): {pers_img[99, 0]} (Red/Magenta)")
    print(f"Bottom-right (99,99): {pers_img[99, 99]} (Yellow/White)")
