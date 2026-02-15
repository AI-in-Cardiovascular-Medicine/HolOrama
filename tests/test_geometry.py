import pytest
import numpy as np
from gui.utils.geometry import SplineGeometry

class TestSplineSplitting:
    
    @pytest.fixture
    def closed_square(self):
        """Creates a 4-point closed square spline."""
        # Square: (0,0), (10,0), (10,10), (0,10)
        x = [0.0, 10.0, 10.0, 0.0]
        y = [0.0, 0.0, 10.0, 10.0]
        return SplineGeometry(
            knot_points_x=x,
            knot_points_y=y,
            start_coords=None,
            end_coords=None,
            n_interpolated_points=100,
            is_closed=True
        )

    def test_split_at_two_indices_counts(self, closed_square):
        """Verify the number of points in each section after a split."""
        # Split at index 1 and 3
        part1, part2 = closed_square.split_at_two_indices(1, 3)
        
        # Part 1 (inside 1 to 3): indices [1, 2, 3] -> 3 points
        assert len(part1.knot_points_x) == 3
        # Part 2 (wrap 3 to 1): indices [3, 0, 1] -> 3 points
        assert len(part2.knot_points_x) == 3
        
        assert part1.is_closed is False
        assert part2.is_closed is False

    def test_split_stitch_roundtrip(self, closed_square):
        """Verify that splitting and stitching restores the original geometry."""
        # 1. Split the square
        part1, part2 = closed_square.split_at_two_indices(1, 3)
        
        # 2. Stitch them back together
        # Note: part1 ends at index 3 (0, 10), part2 starts at index 3 (0, 10)
        reconstructed = part1.stitch_with(part2, close_final=True)
        
        # 3. Check properties
        assert reconstructed.is_closed is True
        
        # The points might be shifted in order, but the cycle should be identical.
        # Original: (0,0), (10,0), (10,10), (0,10)
        # Reconstructed might start at (10,0) depending on which part was 'self'
        assert len(reconstructed.knot_points_x) == len(closed_square.knot_points_x)
        
        # Check if the total area/shape is preserved by checking point existence
        original_points = set(zip(closed_square.knot_points_x, closed_square.knot_points_y))
        new_points = set(zip(reconstructed.knot_points_x, reconstructed.knot_points_y))
        assert original_points == new_points

    def test_interpolation_after_split(self, closed_square):
        """Ensure the split parts can still be interpolated without error."""
        part1, _ = closed_square.split_at_two_indices(0, 2)
        
        # Test if scipy handles the open spline interpolation
        ix, iy = part1.interpolate()
        
        assert len(ix) == part1.n_interpolated_points
        assert not np.array_equal(ix, part1.knot_points_x) # Ensure it actually interpolated

    def test_uneven_split_indices(self, closed_square):
        """Test split with indices provided in reverse order (3, 1 instead of 1, 3)."""
        part1_a, part2_a = closed_square.split_at_two_indices(1, 3)
        part1_b, part2_b = closed_square.split_at_two_indices(3, 1)
        
        assert part1_a.knot_points_x == part1_b.knot_points_x
        assert part2_a.knot_points_x == part2_b.knot_points_x