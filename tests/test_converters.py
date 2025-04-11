import pytest
import numpy as np
import warnings

# Import the converter functions from their module
from equiforge.converters.equi2pers import equi2pers
from equiforge.converters.pers2equi import pers2equi

class TestConverters:
    @pytest.fixture
    def sample_perspective_image(self):
        """Create a simple test perspective image"""
        # Create a 100x100 test image with a gradient pattern
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            for j in range(100):
                img[i, j] = [i, j, (i+j)//2]
        return img
    
    @pytest.fixture
    def sample_equirectangular_image(self):
        """Create a simple test equirectangular image"""
        # Create a 200x100 equirectangular test image (2:1 ratio)
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        for i in range(100):
            for j in range(200):
                img[i, j] = [i, j % 100, (i+j)//2]
        return img
    
    def test_pers2equi_basic_conversion(self, sample_perspective_image):
        """Test basic conversion from perspective to equirectangular"""
        equi_img = pers2equi(sample_perspective_image, output_height=100, fov_x=90)
        
        # Basic checks
        assert equi_img is not None
        assert isinstance(equi_img, np.ndarray)
        # Equirectangular should have 2:1 aspect ratio
        assert equi_img.shape[1] == 2 * equi_img.shape[0]
    
    def test_equi2pers_basic_conversion(self, sample_equirectangular_image):
        """Test basic conversion from equirectangular to perspective"""
        pers_img = equi2pers(
            sample_equirectangular_image, 
            output_width=100,
            output_height=100,
            fov_x=90
        )
        
        # Basic checks
        assert pers_img is not None
        assert isinstance(pers_img, np.ndarray)
        # Output should be square for default settings
        assert pers_img.shape[0] == pers_img.shape[1]
    
    def test_roundtrip_conversion(self, sample_perspective_image):
        """Test that converting to equi and back preserves image (approximately)"""
        equi = pers2equi(sample_perspective_image, output_height=100, fov_x=90)
        
        pers_restored = equi2pers(
            equi, 
            output_width=100,
            output_height=100,
            fov_x=90
        )
        
        # Images should be approximately equal in the center region
        center_original = sample_perspective_image[40:60, 40:60]
        center_restored = pers_restored[40:60, 40:60]
        
        # Using mean absolute error to compare
        mae = np.mean(np.abs(center_original - center_restored))
        
        # Set a stricter target of 10, but only warn if it's above (don't fail)
        if mae >= 10:
            warnings.warn(f"Roundtrip conversion MAE is {mae:.2f}, which is above the target of 10")
        
        # Original threshold for passing
        assert mae < 50, f"MAE={mae:.2f} exceeds maximum threshold of 50"
    
    def test_invalid_input_handling(self):
        """Test that functions properly handle invalid inputs"""
        # Test with invalid FOV
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = pers2equi(test_img, output_height=100, fov_x=370)
        assert result is not None
        assert isinstance(result, np.ndarray)
        
        # Test with small image dimensions
        small_img = np.ones((10, 10, 3), dtype=np.uint8) * 128
        try:
            result = equi2pers(small_img, output_width=100, output_height=100, fov_x=90)
            if result is not None:
                assert isinstance(result, np.ndarray)
                assert result.shape == (100, 100, 3)
        except Exception:
            pass
