import pytest
import numpy as np
# Import the converter functions directly from their modules
from equiforge.converters.equi2pers import equi2pers
from equiforge.converters.pers2equi import pers2equi
from equiforge.utils import projection_utils

class TestIntegration:
    @pytest.fixture
    def real_world_image(self):
        """Load a real test image or create a realistic test pattern"""
        # In real tests, you might load an actual image file
        # Here we create a synthetic one
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        # Create a pattern that makes visual sense (like a grid)
        for i in range(512):
            for j in range(512):
                if (i // 32) % 2 == (j // 32) % 2:
                    img[i, j] = [200, 200, 200]
                else:
                    img[i, j] = [100, 100, 100]
        return img
    
    def test_full_pipeline(self, real_world_image):
        """Test a complete processing pipeline"""
        # Disable GPU for CI workflow compatibility
        equi = pers2equi(real_world_image, output_height=256, fov_x=120, use_gpu=False, )
        
        assert equi is not None, "pers2equi returned None"
        
        # Step 2: Generate multiple perspective views
        views = []
        for angle in [0, 90, 180, 270]:
            # Use correct parameter names - yaw in degrees for the API
            view = equi2pers(
                equi, 
                output_width=256, 
                output_height=256, 
                fov_x=90, 
                yaw=angle,  # Use degrees directly
                use_gpu=False
            )
            
            assert view is not None, f"equi2pers returned None for angle {angle}"
            views.append(view)
            
        # Verify we have the expected number of output views
        assert len(views) == 4
        
        # All views should have the same dimensions
        for view in views:
            assert view.shape == (256, 256, 3), f"View has incorrect shape: {view.shape}"
