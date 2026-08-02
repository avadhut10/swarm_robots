"""
Map merger using weighted average merging and optional ICP alignment
"""

import numpy as np
from scipy.spatial import cKDTree

class MapMerger:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.merged_grid = np.zeros((grid_size, grid_size))
        self.confidence_grid = np.zeros((grid_size, grid_size))
        self.quality_score = 0.0
        self.alignment_threshold = 0.5  # meters
        
    def merge_maps(self, local_grids):
        """Merge multiple local grids using weighted averaging"""
        if not local_grids:
            return
            
        # Reset merged grid
        self.merged_grid = np.zeros((self.grid_size, self.grid_size))
        total_weights = np.zeros((self.grid_size, self.grid_size))
        
        # Weighted average based on observation count
        for grid in local_grids:
            if grid is None:
                continue
                
            # Calculate weights based on log-odds magnitude (certainty)
            weights = np.abs(grid)
            
            # Add weighted contribution
            self.merged_grid += grid * weights
            total_weights += weights
            
        # Avoid division by zero
        mask = total_weights > 0
        self.merged_grid[mask] /= total_weights[mask]
        
        # Update quality score based on consistency
        self._calculate_quality_score(local_grids)
        
    def _calculate_quality_score(self, local_grids):
        """Calculate map merge quality based on overlap consistency"""
        if len(local_grids) < 2:
            self.quality_score = 1.0
            return
            
        consistency_scores = []
        
        for i in range(len(local_grids)):
            for j in range(i + 1, len(local_grids)):
                if local_grids[i] is None or local_grids[j] is None:
                    continue
                    
                # Find overlapping regions (where both grids have observations)
                overlap = (np.abs(local_grids[i]) > 0.5) & (np.abs(local_grids[j]) > 0.5)
                
                if np.sum(overlap) > 0:
                    # Calculate correlation in overlapping region
                    grid_i_overlap = local_grids[i][overlap]
                    grid_j_overlap = local_grids[j][overlap]
                    
                    correlation = np.corrcoef(grid_i_overlap, grid_j_overlap)[0, 1]
                    consistency_scores.append(max(0, correlation))
                    
        if consistency_scores:
            self.quality_score = np.mean(consistency_scores)
        else:
            self.quality_score = 0.0
            
    def align_grids(self, source_grid, target_grid):
        """Align source grid to target grid using ICP-like approach"""
        # This is a simplified 2D ICP implementation
        # Extract occupied cells from both grids
        source_points = self._extract_occupied_points(source_grid)
        target_points = self._extract_occupied_points(target_grid)
        
        if len(source_points) < 10 or len(target_points) < 10:
            return source_grid  # Not enough points for alignment
            
        # Simple centroid-based alignment
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        translation = target_centroid - source_centroid
        
        # Apply translation (simple shift)
        aligned_grid = np.roll(source_grid, int(translation[0]), axis=0)
        aligned_grid = np.roll(aligned_grid, int(translation[1]), axis=1)
        
        return aligned_grid
        
    def _extract_occupied_points(self, grid):
        """Extract coordinates of occupied cells from grid"""
        # Threshold to binary occupancy
        binary_grid = grid > 0.5
        
        # Get coordinates of occupied cells
        points = np.argwhere(binary_grid)
        
        return points
        
    def get_coverage(self):
        """Calculate total coverage percentage of merged map"""
        explored = np.abs(self.merged_grid) > 0.5
        return np.sum(explored) / (self.grid_size * self.grid_size)
        
    def get_merged_grid(self):
        """Return the merged grid"""
        return self.merged_grid
        
    def get_confidence_grid(self):
        """Return the confidence grid showing observation density"""
        return self.confidence_grid