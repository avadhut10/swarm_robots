import json
import math
import os
from datetime import datetime

class WorkspaceValidator:
    """
    Validates the workspace map by calculating distances between markers
    Uses world coordinates (mm) if available, otherwise pixel coordinates
    """
    
    def __init__(self, map_file=None):
        """
        Initialize validator
        
        Args:
            map_file: Path to your workspace_map_*.json file
        """
        self.map_file = map_file
        self.map_data = None
        self.positions = {}
        self.workspace_width = 0
        self.workspace_height = 0
        self.marker_sizes = {}
        self.boundary_set = False
        self.coordinate_type = "pixel"  # Will be updated
        
        if map_file and os.path.exists(map_file):
            self.load_map(map_file)
        else:
            # Try to find the map file automatically
            import glob
            map_files = glob.glob("workspace_map_*.json")
            if map_files:
                self.load_map(sorted(map_files)[-1])
            else:
                print(" No map file found!")
                print("   Please run the real-time detector and save a map first")
    
    def load_map(self, map_file):
        """Load the workspace map from JSON"""
        try:
            with open(map_file, 'r') as f:
                self.map_data = json.load(f)
            
            self.positions = self.map_data.get('positions', {})
            self.workspace_width = self.map_data.get('workspace_size_mm', [2100, 2950])[0]
            self.workspace_height = self.map_data.get('workspace_size_mm', [2100, 2950])[1]
            self.marker_sizes = self.map_data.get('marker_sizes_mm', {})
            self.boundary_set = self.map_data.get('boundary_set', False)
            
            # Check if this is pixel or mm coordinates
            # If coordinates are > 5000, they're likely pixels
            sample_pos = next(iter(self.positions.values())) if self.positions else None
            if sample_pos:
                x = sample_pos.get('x_mm', 0)
                y = sample_pos.get('y_mm', 0)
                if x > 5000 or y > 5000:
                    self.coordinate_type = "pixel"
                    print("   Coordinates are in PIXELS (need to convert to mm)")
                else:
                    self.coordinate_type = "mm"
                    print("   Coordinates are in MILLIMETERS")
            
            print(f" Map loaded: {map_file}")
            print(f"   Workspace: {self.workspace_width}x{self.workspace_height}mm")
            print(f"   Markers found: {len(self.positions)}")
            print(f"   Boundary set: {'Yes' if self.boundary_set else 'No'}")
            print(f"   Coordinate type: {self.coordinate_type.upper()}")
            return True
            
        except Exception as e:
            print(f" Could not load map: {e}")
            return False
    
    def calculate_distance(self, pos1, pos2):
        """Calculate distance between two points in mm"""
        return math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    
    def get_marker_position(self, marker_id):
        """Get position of a marker by ID"""
        marker_id_str = str(marker_id)
        if marker_id_str in self.positions:
            return (self.positions[marker_id_str]['x_mm'], 
                    self.positions[marker_id_str]['y_mm'])
        return None
    
    def get_marker_label(self, marker_id):
        """Get label of a marker by ID"""
        marker_id_str = str(marker_id)
        if marker_id_str in self.positions:
            return self.positions[marker_id_str]['label']
        return f"ID{marker_id}"
    
    def validate_workspace(self):
        """Validate workspace dimensions using corner markers"""
        print("\n" + "="*60)
        print("WORKSPACE DIMENSION VALIDATION")
        print("="*60)
        
        # Get corner positions
        corners = {}
        for id in [0, 1, 2, 3]:
            pos = self.get_marker_position(id)
            if pos:
                corners[id] = pos
        
        if len(corners) != 4:
            print(f" Only {len(corners)}/4 corner markers found!")
            print("   Make sure all corner markers (IDs 0-3) are detected")
            return False
        
        # Calculate distances
        distances = {
            'Top (0 to 1)': self.calculate_distance(corners[0], corners[1]),
            'Bottom (2 to 3)': self.calculate_distance(corners[2], corners[3]),
            'Left (0 to 2)': self.calculate_distance(corners[0], corners[2]),
            'Right (1 to 3)': self.calculate_distance(corners[1], corners[3]),
            'Diagonal (0 to 3)': self.calculate_distance(corners[0], corners[3]),
            'Diagonal (1 to 2)': self.calculate_distance(corners[1], corners[2]),
        }
        
        # Expected values
        expected_width = self.workspace_width
        expected_height = self.workspace_height
        expected_diagonal = math.sqrt(expected_width**2 + expected_height**2)
        
        # Print results
        print(f"\nExpected workspace: {expected_width} x {expected_height} mm")
        print(f"Expected diagonal: {expected_diagonal:.1f} mm")
        print(f"Coordinate type: {self.coordinate_type.upper()}")
        print("\nMEASURED DISTANCES:")
        print("-"*50)
        
        errors = {}
        for name, dist in distances.items():
            # Determine expected value for this measurement
            if 'Top' in name or 'Bottom' in name:
                expected = expected_width
            elif 'Left' in name or 'Right' in name:
                expected = expected_height
            else:
                expected = expected_diagonal
            
            error = abs(dist - expected)
            errors[name] = error
            
            # Determine status
            if error < 25:
                status = "GOOD"
            elif error < 50:
                status = "FAIR"
            else:
                status = "POOR"
            
            print(f"{name}: {dist:.1f} mm (Error: {error:.1f} mm) [{status}]")
        
        # Summary
        avg_error = sum(errors.values()) / len(errors)
        max_error = max(errors.values())
        min_error = min(errors.values())
        
        print("\n" + "-"*50)
        print("SUMMARY:")
        print(f"   Average error: {avg_error:.1f} mm")
        print(f"   Maximum error: {max_error:.1f} mm")
        print(f"   Minimum error: {min_error:.1f} mm")
        
        # Quality assessment
        if avg_error < 25:
            print("   EXCELLENT! System is very accurate.")
            return True
        elif avg_error < 50:
            print("   GOOD! System is accurate enough for most tasks.")
            return True
        elif avg_error < 100:
            print("   FAIR - Consider re-calibrating for better accuracy.")
            return False
        else:
            print("   POOR - Need to re-calibrate!")
            print("   The coordinates appear to be in pixels, not mm.")
            return False
    
    def show_all_markers(self):
        """Show all detected markers and their positions"""
        print("\n" + "="*60)
        print("ALL DETECTED MARKERS")
        print("="*60)
        
        # Group markers by type
        groups = {
            'Boundary': [0, 1, 2, 3],
            'Start/End': [10, 11],
            'Jobs': [20, 21],
            'Robots': [100, 101, 102]
        }
        
        for group_name, ids in groups.items():
            print(f"\n{group_name}:")
            found = False
            for id in ids:
                pos = self.get_marker_position(id)
                if pos:
                    found = True
                    label = self.get_marker_label(id)
                    print(f"   {label} (ID {id}): ({pos[0]:.1f}, {pos[1]:.1f}) {self.coordinate_type}")
                else:
                    print(f"   ID {id}: NOT DETECTED")
            if not found:
                print(f"   None detected")
    
    def calculate_marker_distances(self):
        """Calculate distances between all markers"""
        print("\n" + "="*60)
        print("DISTANCES BETWEEN MARKERS")
        print("="*60)
        
        # Important marker pairs
        pairs = [
            (10, 20, "START -> JOB 1"),
            (10, 21, "START -> JOB 2"),
            (10, 11, "START -> END"),
            (20, 21, "JOB 1 -> JOB 2"),
            (20, 100, "JOB 1 -> ROBOT 1"),
            (20, 101, "JOB 1 -> ROBOT 2"),
            (20, 102, "JOB 1 -> ROBOT 3"),
            (21, 100, "JOB 2 -> ROBOT 1"),
            (21, 101, "JOB 2 -> ROBOT 2"),
            (21, 102, "JOB 2 -> ROBOT 3"),
            (100, 101, "ROBOT 1 -> ROBOT 2"),
            (100, 102, "ROBOT 1 -> ROBOT 3"),
            (101, 102, "ROBOT 2 -> ROBOT 3"),
        ]
        
        print("\nMarker Distances:")
        print("-"*50)
        
        for id1, id2, label in pairs:
            pos1 = self.get_marker_position(id1)
            pos2 = self.get_marker_position(id2)
            
            if pos1 and pos2:
                dist = self.calculate_distance(pos1, pos2)
                print(f"{label}: {dist:.1f} mm")
            else:
                missing = []
                if not pos1:
                    missing.append(f"ID{id1}")
                if not pos2:
                    missing.append(f"ID{id2}")
                print(f"{label}: {', '.join(missing)} not detected")
    
    def save_validation_report(self):
        """Save validation results to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"workspace_validation_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("WORKSPACE VALIDATION REPORT\n")
                f.write("="*60 + "\n\n")
                
                f.write(f"Map file: {self.map_file}\n")
                f.write(f"Workspace: {self.workspace_width}x{self.workspace_height}mm\n")
                f.write(f"Markers found: {len(self.positions)}\n")
                f.write(f"Coordinate type: {self.coordinate_type.upper()}\n")
                f.write(f"Boundary set: {'Yes' if self.boundary_set else 'No'}\n\n")
                
                f.write("ALL MARKER POSITIONS:\n")
                f.write("-"*40 + "\n")
                
                for marker_id_str, data in self.positions.items():
                    f.write(f"  {data['label']} (ID {marker_id_str}): ")
                    f.write(f"({data['x_mm']:.1f}, {data['y_mm']:.1f}) {self.coordinate_type}\n")
                
                f.write("\nVALIDATION RESULT:\n")
                f.write("-"*40 + "\n")
                
                # Calculate corner distances
                corners = {}
                for id in [0, 1, 2, 3]:
                    pos = self.get_marker_position(id)
                    if pos:
                        corners[id] = pos
                
                if len(corners) == 4:
                    distances = {
                        'Top (0 to 1)': self.calculate_distance(corners[0], corners[1]),
                        'Bottom (2 to 3)': self.calculate_distance(corners[2], corners[3]),
                        'Left (0 to 2)': self.calculate_distance(corners[0], corners[2]),
                        'Right (1 to 3)': self.calculate_distance(corners[1], corners[3]),
                    }
                    
                    for name, dist in distances.items():
                        f.write(f"{name}: {dist:.1f} mm\n")
            
            print(f"\n Validation report saved to: {filename}")
            
        except Exception as e:
            print(f" Could not save report: {e}")
            # Try saving without special characters
            try:
                filename_ascii = f"workspace_validation_{timestamp}_ascii.txt"
                with open(filename_ascii, 'w') as f:
                    f.write("WORKSPACE VALIDATION REPORT\n")
                    f.write("="*60 + "\n\n")
                    f.write(f"Map file: {self.map_file}\n")
                    f.write(f"Workspace: {self.workspace_width}x{self.workspace_height}mm\n")
                    f.write(f"Markers found: {len(self.positions)}\n")
                print(f" Simple report saved to: {filename_ascii}")
            except:
                print(" Could not save report")
    
    def run_validation(self):
        """Run full validation"""
        if not self.map_data:
            print(" No map loaded!")
            return
        
        print("\n" + "="*60)
        print("RUNNING WORKSPACE VALIDATION")
        print("="*60)
        
        # 1. Show all markers
        self.show_all_markers()
        
        # 2. Validate workspace dimensions
        is_valid = self.validate_workspace()
        
        # 3. Calculate marker distances
        self.calculate_marker_distances()
        
        # 4. Save report
        self.save_validation_report()
        
        # Final verdict
        print("\n" + "="*60)
        if is_valid:
            print(" VALIDATION PASSED! System is ready.")
            print(" Next step: Simulation or Robot Control")
        else:
            print(" VALIDATION WARNING - Some measurements are off.")
            print(" Recommended Actions:")
            if self.coordinate_type == "pixel":
                print("   - The map saved PIXEL coordinates, not mm coordinates")
                print("   - Re-run the real-time detector with calibration")
                print("   - Make sure the perspective transform is working")
            else:
                print("   - Re-calibrate camera")
                print("   - Check marker placement")
                print("   - Re-run the real-time detector")
        print("="*60)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("WORKSPACE VALIDATOR")
    print("="*60)
    
    # Find the latest workspace map
    import glob
    map_files = glob.glob("F:\swarm_robots\codes\computer_vision\path_planning\workspace_map_20260703_131010.json")
    
    if not map_files:
        print(" No workspace map found!")
        print("   Please run the real-time detector and save a map first")
        exit()
    
    # Use the latest map
    map_file = sorted(map_files)[-1]
    print(f"Using map: {map_file}")
    
    # Create and run validator
    validator = WorkspaceValidator(map_file)
    validator.run_validation()