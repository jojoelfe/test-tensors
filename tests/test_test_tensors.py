import numpy as np
import pytest

import test_tensors
from test_tensors import generate_cross_3d


def test_imports_with_version():
    assert isinstance(test_tensors.__version__, str)


def test_visualization_import():
    """Test that visualization functions can be imported."""
    # This should work even without viz dependencies
    from test_tensors import visualize_3d_tensor, create_demo_notebook
    
    assert callable(visualize_3d_tensor)
    assert callable(create_demo_notebook)


class TestGenerateCross3D:
    """Test the generate_cross_3d function."""

    def test_default_shape(self):
        """Test default cubic shape."""
        cross = generate_cross_3d()
        assert cross.shape == (64, 64, 64)
        assert cross.dtype == np.float64

    def test_cubic_shape_int(self):
        """Test cubic shape from single integer."""
        cross = generate_cross_3d(32)
        assert cross.shape == (32, 32, 32)
        assert cross.dtype == np.float64

    def test_rectangular_shape_tuple(self):
        """Test rectangular shape from tuple."""
        cross = generate_cross_3d((10, 20, 30))
        assert cross.shape == (10, 20, 30)
        assert cross.dtype == np.float64

    def test_invalid_shape_tuple_length(self):
        """Test that invalid tuple length raises error."""
        with pytest.raises(ValueError, match="Shape must be int or tuple of 3 ints"):
            generate_cross_3d((10, 20))

        with pytest.raises(ValueError, match="Shape must be int or tuple of 3 ints"):
            generate_cross_3d((10, 20, 30, 40))

    def test_cross_pattern_cubic(self):
        """Test that cross pattern is correct for cubic volume."""
        cross = generate_cross_3d(10)
        center = 5

        # Check center point is part of cross
        assert cross[center, center, center] == 1.0

        # Check cross lines are present
        # XY plane cross at center Z
        assert np.all(cross[center, :, center] == 1.0)  # Horizontal line
        assert np.all(cross[center, center, :] == 1.0)  # Vertical line

        # Z direction line through center
        assert np.all(cross[:, center, center] == 1.0)

        # Check background is zero
        # Pick a corner that should be zero
        assert cross[0, 0, 0] == 0.0
        assert cross[9, 9, 9] == 0.0

    def test_cross_pattern_rectangular(self):
        """Test that cross pattern is correct for rectangular volume."""
        cross = generate_cross_3d((8, 12, 16))
        center_z, center_y, center_x = 4, 6, 8

        # Check center point is part of cross
        assert cross[center_z, center_y, center_x] == 1.0

        # Check cross lines are present
        assert np.all(cross[center_z, :, center_x] == 1.0)  # Y direction
        assert np.all(cross[center_z, center_y, :] == 1.0)  # X direction
        assert np.all(cross[:, center_y, center_x] == 1.0)  # Z direction

        # Check some background points are zero
        assert cross[0, 0, 0] == 0.0
        assert cross[7, 11, 15] == 0.0

    def test_cross_structure_properties(self):
        """Test structural properties of the cross."""
        cross = generate_cross_3d(20)

        # Count non-zero elements (should be cross structure)
        non_zero_count = np.count_nonzero(cross)

        # Expected: 3 orthogonal lines through center
        # Each line has 20 elements, but they intersect at center
        # Line 1: (center, :, center) = 20 elements
        # Line 2: (center, center, :) = 20 elements
        # Line 3: (:, center, center) = 20 elements
        # Overlaps: center point counted 3 times, need to subtract 2
        # XY plane intersections already counted in lines 1&2
        expected_count = 20 + 20 + 20 - 2  # Subtract double-counted center
        assert non_zero_count == expected_count

        # Check that all non-zero values are 1.0
        non_zero_values = cross[cross != 0]
        assert np.all(non_zero_values == 1.0)

    def test_different_sizes(self):
        """Test various sizes work correctly."""
        for size in [1, 3, 5, 10, 17, 64, 100]:
            cross = generate_cross_3d(size)
            assert cross.shape == (size, size, size)

            center = size // 2
            # Center should always be part of cross
            assert cross[center, center, center] == 1.0


class TestVisualizationFunctions:
    """Test visualization functionality (without requiring viz dependencies)."""

    def test_visualization_import_error_without_deps(self):
        """Test that proper error is raised when viz deps are missing."""
        from test_tensors import visualize_3d_tensor, create_demo_notebook
        
        cross = generate_cross_3d(10)
        
        # These should raise ImportError if viz dependencies aren't available
        # In CI/testing environments without viz deps installed
        try:
            visualize_3d_tensor(cross, show_3d=False)
        except ImportError as e:
            assert "visualization requires" in str(e).lower()
        
        try:
            create_demo_notebook()
        except ImportError as e:
            assert "requires optional dependencies" in str(e).lower()

    def test_visualization_validation(self):
        """Test input validation for visualization functions."""
        from test_tensors import visualize_3d_tensor
        
        # Test with invalid input dimensions
        with pytest.raises((ValueError, ImportError)):
            # 2D array should fail validation (or ImportError if no viz deps)
            visualize_3d_tensor(np.ones((10, 10)))
        
        with pytest.raises((ValueError, ImportError)):
            # 4D array should fail validation (or ImportError if no viz deps)  
            visualize_3d_tensor(np.ones((5, 5, 5, 5)))
        
        with pytest.raises((ValueError, ImportError)):
            # Empty array should fail validation (or ImportError if no viz deps)
            visualize_3d_tensor(np.array([]))
