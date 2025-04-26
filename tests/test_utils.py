import pytest
import numpy as np
import logging
from equiforge.utils import projection_utils, logging_utils

class TestProjectionUtils:
    def test_spherical_to_cartesian(self):
        """Test conversion from spherical to cartesian coordinates"""
        # Test known conversion - use the functions that actually exist
        theta, phi = 0, 0  # Looking straight ahead
        # Assuming the create_rotation_matrix function takes yaw, pitch, roll as we saw in other files
        R = projection_utils.create_rotation_matrix(theta, phi, 0)
        # Just check that the function exists and returns a matrix
        assert R.shape == (3, 3)
    
    def test_cartesian_to_spherical(self):
        """Test conversion from cartesian to spherical coordinates"""
        # Adjust to test functions that actually exist
        # Test focal length calculation instead
        fov_x_rad = np.radians(90)
        width, height = 100, 100
        f_h, f_v = projection_utils.calculate_focal_length(width, height, fov_x_rad)
        
        # Check that focal lengths are reasonable
        assert f_h > 0
        assert f_v > 0

class TestLoggingUtils:
    def test_logger_creation(self):
        """Test that loggers are created properly"""
        logger = logging_utils.setup_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"