import cv2
import numpy as np
import time
import math
import json
from datetime import datetime
import os

class CalibrationValidator:
    """
    Validates camera calibration and distance measurements
    Uses corner markers to create a verification grid
    """
    
    def __init__(self, camera_id=1, calibration_file=None):
        """
        Initialize validation system
        
        Args:
            camera_id: Camera device ID
            calibration_file: Path to calibration .npz file
        """
        self.camera_id = camera_id
        self.cap = None
        self.is_camera_open = False
        
        # Load calibration
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.is_calibrated = False
        self.calibration_error = 0
        
        if calibration_file and os.path.exists(calibration_file):
            self.load_calibration(calibration_file)
        
        # ArUco setup
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        except:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5)
        
        self.parameters = cv2.aruco.DetectorParameters()
        
        # ============================================================
        # YOUR ACTUAL WORKSPACE AND MARKER MEASUREMENTS
        # ============================================================
        self.workspace_width_mm = 2800
        self.workspace_height_mm = 2200
        
        self.marker_sizes_mm = {
            0: 47, 1: 47, 2: 47, 3: 47,
            10: 45, 11: 45,
            20: 33.5, 21: 33.5,
            100: 32, 101: 32, 102: 32,
        }
        
        # Perspective transform
        self.perspective_matrix = None
        self.boundary_points = {}
        self.is_workspace_setup = False
        self.inverse_perspective = None
        
        # Measurement data
        self.measured_distances = {}
        self.grid_points = []
        self.accuracy_errors = []
        
        # Camera height estimation
        self.camera_height_mm = None
        self.height_measurements = []
        
        # Detection data
        self.detected_markers = {}
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        print("📐 Calibration Validation System")
        print("="*60)
        print(f"Workspace: {self.workspace_width_mm}x{self.workspace_height_mm}mm")
        print(f"Calibration: {'✅ Loaded' if self.is_calibrated else '❌ Not loaded'}")
        print("="*60)
    
    def load_calibration(self, filename):
        """Load camera calibration"""
        try:
            data = np.load(filename)
            self.camera_matrix = data['camera_matrix']
            self.distortion_coeffs = data['distortion_coeffs']
            self.is_calibrated = True
            if 'calibration_error' in data:
                self.calibration_error = data['calibration_error']
            print(f"✅ Calibration loaded: {filename}")
            print(f"   Reprojection error: {self.calibration_error:.4f} pixels")
            return True
        except Exception as e:
            print(f"❌ Could not load calibration: {e}")
            return False
    
    def open_camera(self):
        """Open Motorola Smart Connect camera"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"❌ Could not open camera {self.camera_id}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.is_camera_open = True
        print(f"✅ Camera opened: {width}x{height}")
        return True
    
    def close_camera(self):
        """Close camera"""
        if self.cap is not None:
            self.cap.release()
            self.is_camera_open = False
    
    def undistort_frame(self, frame):
        """Apply camera calibration"""
        if not self.is_calibrated or self.camera_matrix is None:
            return frame
        
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.distortion_coeffs, (w, h), 1, (w, h)
        )
        
        undistorted = cv2.undistort(
            frame, self.camera_matrix, self.distortion_coeffs, None, new_camera_matrix
        )
        
        x, y, w, h = roi
        if x > 0 and y > 0 and w > 0 and h > 0:
            undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted
    
    def detect_markers(self, frame):
        """Detect ArUco markers"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        try:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.parameters
            )
        except:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict
            )
        
        self.detected_markers = {}
        
        if ids is not None and len(ids) > 0:
            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                marker_corners = corners[i][0]
                
                center_x = int(np.mean(marker_corners[:, 0]))
                center_y = int(np.mean(marker_corners[:, 1]))
                
                side1 = np.linalg.norm(marker_corners[0] - marker_corners[1])
                side2 = np.linalg.norm(marker_corners[1] - marker_corners[2])
                pixel_size = (side1 + side2) / 2
                
                dx = marker_corners[1][0] - marker_corners[0][0]
                dy = marker_corners[1][1] - marker_corners[0][1]
                angle = np.arctan2(dy, dx) * 180 / np.pi
                
                self.detected_markers[marker_id] = {
                    'id': marker_id,
                    'corners': marker_corners,
                    'center': (center_x, center_y),
                    'pixel_size': pixel_size,
                    'size_mm': self.marker_sizes_mm.get(marker_id, 0),
                    'angle': angle
                }
        
        return self.detected_markers
    
    def setup_workspace(self, frame):
        """Setup workspace using corner markers"""
        self.detect_markers(frame)
        
        boundary_ids = [0, 1, 2, 3]
        boundary_points = []
        
        for marker_id in boundary_ids:
            if marker_id in self.detected_markers:
                center = self.detected_markers[marker_id]['center']
                boundary_points.append((marker_id, center))
                self.boundary_points[marker_id] = center
        
        if len(boundary_points) == 4:
            ordered_points = []
            for id in [0, 1, 2, 3]:
                if id in self.boundary_points:
                    ordered_points.append(self.boundary_points[id])
            
            if len(ordered_points) == 4:
                src_points = np.array(ordered_points, dtype=np.float32)
                dst_points = np.array([
                    [0, 0],
                    [self.workspace_width_mm, 0],
                    [0, self.workspace_height_mm],
                    [self.workspace_width_mm, self.workspace_height_mm]
                ], dtype=np.float32)
                
                self.perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                self.inverse_perspective = cv2.getPerspectiveTransform(dst_points, src_points)
                self.is_workspace_setup = True
                
                return True
        
        return False
    
    def get_world_coordinates(self, pixel_x, pixel_y):
        """Convert pixel to world coordinates"""
        if self.perspective_matrix is None:
            return pixel_x, pixel_y
        
        pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        world_point = cv2.perspectiveTransform(pixel_point, self.perspective_matrix)
        return world_point[0][0][0], world_point[0][0][1]
    
    def calculate_distance(self, pos1, pos2):
        """Calculate distance between two points in mm"""
        return math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    
    def validate_corner_distances(self):
        """
        Measure distances between corner markers
        Should match your workspace dimensions
        """
        print("\n" + "="*60)
        print("📏 CORNER DISTANCE VALIDATION")
        print("="*60)
        
        # Get corner positions
        corners = {}
        for marker_id in [0, 1, 2, 3]:
            if marker_id in self.detected_markers:
                center = self.detected_markers[marker_id]['center']
                world_x, world_y = self.get_world_coordinates(center[0], center[1])
                corners[marker_id] = (world_x, world_y)
        
        if len(corners) != 4:
            print("❌ All 4 corner markers not detected!")
            return False
        
        # Expected distances (your workspace dimensions)
        expected_width = self.workspace_width_mm
        expected_height = self.workspace_height_mm
        expected_diagonal = math.sqrt(expected_width**2 + expected_height**2)
        
        # Measure actual distances
        distances = {
            'top': self.calculate_distance(corners[0], corners[1]),
            'bottom': self.calculate_distance(corners[2], corners[3]),
            'left': self.calculate_distance(corners[0], corners[2]),
            'right': self.calculate_distance(corners[1], corners[3]),
            'diagonal1': self.calculate_distance(corners[0], corners[3]),
            'diagonal2': self.calculate_distance(corners[1], corners[2]),
        }
        
        # Calculate errors
        errors = {
            'top': abs(distances['top'] - expected_width),
            'bottom': abs(distances['bottom'] - expected_width),
            'left': abs(distances['left'] - expected_height),
            'right': abs(distances['right'] - expected_height),
            'diagonal1': abs(distances['diagonal1'] - expected_diagonal),
            'diagonal2': abs(distances['diagonal2'] - expected_diagonal),
        }
        
        # Print results
        print(f"\n📐 Expected workspace: {expected_width} x {expected_height} mm")
        print(f"   Expected diagonal: {expected_diagonal:.1f} mm")
        print("\n📊 MEASURED DISTANCES:")
        print("-"*50)
        print(f"Top edge (0→1):      {distances['top']:.1f} mm  (Error: {errors['top']:.1f} mm)")
        print(f"Bottom edge (2→3):   {distances['bottom']:.1f} mm  (Error: {errors['bottom']:.1f} mm)")
        print(f"Left edge (0→2):     {distances['left']:.1f} mm  (Error: {errors['left']:.1f} mm)")
        print(f"Right edge (1→3):    {distances['right']:.1f} mm  (Error: {errors['right']:.1f} mm)")
        print(f"Diagonal 1 (0→3):    {distances['diagonal1']:.1f} mm  (Error: {errors['diagonal1']:.1f} mm)")
        print(f"Diagonal 2 (1→2):    {distances['diagonal2']:.1f} mm  (Error: {errors['diagonal2']:.1f} mm)")
        print("-"*50)
        
        # Calculate average error
        avg_error = sum(errors.values()) / len(errors)
        max_error = max(errors.values())
        
        print(f"\n📊 SUMMARY:")
        print(f"   Average error: {avg_error:.1f} mm")
        print(f"   Maximum error: {max_error:.1f} mm")
        
        # Store for later
        self.measured_distances = distances
        self.accuracy_errors = errors
        
        # Quality assessment
        if avg_error < 10:
            print(f"   ✅ EXCELLENT! Error < 10mm")
        elif avg_error < 25:
            print(f"   ✅ GOOD! Error < 25mm")
        elif avg_error < 50:
            print(f"   ⚠️ FAIR - Consider re-calibrating")
        else:
            print(f"   ❌ POOR - Need to re-calibrate!")
        
        return True
    
    def calculate_camera_height(self):
        """
        Calculate camera height from the floor using corner markers
        """
        print("\n" + "="*60)
        print("📐 CAMERA HEIGHT ESTIMATION")
        print("="*60)
        
        if not self.is_calibrated:
            print("❌ Cannot calculate height without camera calibration")
            return None
        
        if len(self.detected_markers) < 4:
            print("❌ Need all 4 corner markers detected")
            return None
        
        heights = []
        
        for marker_id in [0, 1, 2, 3]:
            if marker_id not in self.detected_markers:
                continue
            
            marker = self.detected_markers[marker_id]
            marker_size_mm = marker['size_mm']
            pixel_size = marker['pixel_size']
            
            if marker_size_mm > 0 and pixel_size > 0:
                fx = self.camera_matrix[0, 0]
                height = (marker_size_mm * fx) / pixel_size
                heights.append(height)
        
        if heights:
            avg_height = sum(heights) / len(heights)
            self.camera_height_mm = avg_height
            
            print(f"\n📊 Camera height estimated from {len(heights)} markers:")
            print(f"   Average height: {avg_height:.0f} mm ({avg_height/1000:.2f} meters)")
            print(f"   Individual measurements: {[int(h) for h in heights]} mm")
            
            self.height_measurements = heights
            return avg_height
        
        print("❌ Could not calculate camera height")
        return None
    
    def create_verification_grid(self, grid_spacing_mm=500):
        """Create a grid of verification points"""
        print("\n" + "="*60)
        print(f"📊 VERIFICATION GRID (Spacing: {grid_spacing_mm}mm)")
        print("="*60)
        
        if not self.is_workspace_setup:
            print("❌ Workspace not set up!")
            return None, None
        
        grid_points = []
        x_steps = int(self.workspace_width_mm / grid_spacing_mm) + 1
        y_steps = int(self.workspace_height_mm / grid_spacing_mm) + 1
        
        print(f"\n📐 Grid: {x_steps} x {y_steps} = {x_steps * y_steps} points")
        print(f"   Spacing: {grid_spacing_mm}mm")
        
        for i in range(y_steps):
            for j in range(x_steps):
                x = j * grid_spacing_mm
                y = i * grid_spacing_mm
                x = min(x, self.workspace_width_mm)
                y = min(y, self.workspace_height_mm)
                grid_points.append((x, y))
        
        self.grid_points = grid_points
        
        pixel_points = []
        for world_x, world_y in grid_points:
            world_point = np.array([[[world_x, world_y]]], dtype=np.float32)
            pixel_point = cv2.perspectiveTransform(world_point, self.inverse_perspective)
            px, py = pixel_point[0][0]
            pixel_points.append((int(px), int(py)))
        
        print(f"\n📊 Grid generated with {len(grid_points)} points")
        return grid_points, pixel_points
    
    def draw_validation_grid(self, frame, grid_points, pixel_points):
        """Draw verification grid on frame"""
        if not grid_points or not pixel_points:
            return frame
        
        display = frame.copy()
        grid_spacing_mm = 500
        x_steps = int(self.workspace_width_mm / grid_spacing_mm) + 1
        y_steps = int(self.workspace_height_mm / grid_spacing_mm) + 1
        
        # Draw horizontal lines
        for i in range(y_steps):
            y_world = i * grid_spacing_mm
            y_world = min(y_world, self.workspace_height_mm)
            
            line_points = []
            for j in range(x_steps):
                idx = i * x_steps + j
                if idx < len(pixel_points):
                    line_points.append(pixel_points[idx])
            
            if len(line_points) > 1:
                for k in range(len(line_points) - 1):
                    cv2.line(display, line_points[k], line_points[k+1], (100, 100, 255), 1)
                
                if line_points:
                    cv2.putText(display, f"Y={y_world}mm", 
                               (line_points[0][0] + 5, line_points[0][1] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
        
        # Draw vertical lines
        for j in range(x_steps):
            x_world = j * grid_spacing_mm
            x_world = min(x_world, self.workspace_width_mm)
            
            line_points = []
            for i in range(y_steps):
                idx = i * x_steps + j
                if idx < len(pixel_points):
                    line_points.append(pixel_points[idx])
            
            if len(line_points) > 1:
                for k in range(len(line_points) - 1):
                    cv2.line(display, line_points[k], line_points[k+1], (100, 200, 255), 1)
                
                if line_points:
                    cv2.putText(display, f"X={x_world}mm", 
                               (line_points[0][0] + 5, line_points[0][1] + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
        
        # Draw grid points
        for px, py in pixel_points:
            cv2.circle(display, (px, py), 3, (0, 255, 255), -1)
        
        return display
    
    def measure_chessboard_distance(self):
        """
        Measure chessboard distance at different positions
        Uses a separate capture to avoid conflicts
        """
        print("\n" + "="*60)
        print("🎯 CHESSBOARD DISTANCE MEASUREMENT")
        print("="*60)
        print("Place the chessboard at different positions on the floor")
        print("The system will measure the distance from the camera")
        print("Compare this with your actual measured distance")
        print("Press SPACE to capture, 'q' to return to main menu")
        print("="*60)
        
        if not self.is_calibrated:
            print("❌ Camera not calibrated - distance measurements won't be accurate")
            return
        
        # Create a separate capture for chessboard measurement
        chess_cap = cv2.VideoCapture(self.camera_id)
        if not chess_cap.isOpened():
            print("❌ Could not open camera for chessboard measurement")
            return
        
        chess_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        chess_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        measurements = []
        chessboard_size = (8, 5)
        square_size_mm = 30
        
        print("\n📸 Press SPACE when chessboard is detected and you've measured the distance")
        print("   Enter the actual distance in mm when prompted")
        print("   Press 'q' to return to main menu")
        
        while True:
            ret, frame = chess_cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            if self.is_calibrated:
                frame = self.undistort_frame(frame)
            
            # Detect chessboard
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(
                gray, chessboard_size, None,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            display = frame.copy()
            chessboard_detected = False
            estimated_distance = 0
            
            if ret:
                chessboard_detected = True
                cv2.drawChessboardCorners(display, chessboard_size, corners, ret)
                
                # Calculate distance
                side_length_pixels = np.linalg.norm(corners[0][0] - corners[1][0])
                side_length_mm = square_size_mm * chessboard_size[0]
                
                if side_length_pixels > 0 and self.is_calibrated:
                    fx = self.camera_matrix[0, 0]
                    estimated_distance = (side_length_mm * fx) / side_length_pixels
                    
                    cv2.putText(display, f"Estimated Distance: {estimated_distance/1000:.2f} m", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display, "Press SPACE to record this measurement", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(display, "❌ Place chessboard in view", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Chessboard Distance Measurement', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == 32 and chessboard_detected:  # SPACE
                print("\n📊 Measurement captured!")
                print("   Enter the ACTUAL distance you measured with tape measure (in mm):")
                try:
                    actual_input = input("   Actual distance (mm): ").strip()
                    if actual_input == '':
                        print("   ❌ No input - measurement not recorded")
                        continue
                    
                    actual_distance = float(actual_input)
                    
                    if actual_distance <= 0:
                        print("   ❌ Invalid distance - must be positive")
                        continue
                    
                    error = abs(estimated_distance - actual_distance)
                    error_percent = (error / actual_distance) * 100
                    
                    measurements.append({
                        'estimated': estimated_distance,
                        'actual': actual_distance,
                        'error': error,
                        'error_percent': error_percent
                    })
                    
                    print(f"\n   Results:")
                    print(f"   Estimated: {estimated_distance:.0f} mm")
                    print(f"   Actual:    {actual_distance:.0f} mm")
                    print(f"   Error:     {error:.0f} mm ({error_percent:.1f}%)")
                    
                    if error_percent < 5:
                        print("   ✅ Excellent accuracy!")
                    elif error_percent < 10:
                        print("   ✅ Good accuracy!")
                    elif error_percent < 20:
                        print("   ⚠️ Fair accuracy - consider re-calibrating")
                    else:
                        print("   ❌ Poor accuracy - need to re-calibrate!")
                    
                    print(f"\n   Measurements recorded: {len(measurements)}")
                    print("   Continue measuring or press 'q' to return")
                    
                except ValueError:
                    print("   ❌ Invalid input - enter a number (e.g., 1500)")
                except KeyboardInterrupt:
                    print("\n   Measurement cancelled")
                    break
        
        chess_cap.release()
        cv2.destroyWindow('Chessboard Distance Measurement')
        
        # Summary
        if measurements:
            print("\n" + "="*60)
            print("📊 DISTANCE MEASUREMENT SUMMARY")
            print("="*60)
            
            avg_error = sum(m['error'] for m in measurements) / len(measurements)
            avg_error_percent = sum(m['error_percent'] for m in measurements) / len(measurements)
            max_error = max(m['error'] for m in measurements)
            min_error = min(m['error'] for m in measurements)
            
            print(f"\nMeasurements recorded: {len(measurements)}")
            print(f"Average error: {avg_error:.1f} mm ({avg_error_percent:.1f}%)")
            print(f"Min error: {min_error:.1f} mm")
            print(f"Max error: {max_error:.1f} mm")
            
            print("\n" + "-"*40)
            for i, m in enumerate(measurements, 1):
                print(f"  {i}. Estimated: {m['estimated']:.0f}mm | Actual: {m['actual']:.0f}mm | Error: {m['error']:.0f}mm ({m['error_percent']:.1f}%)")
            
            print("\n" + "-"*40)
            if avg_error_percent < 5:
                print("✅ System is ACCURATE! Ready for robot operations.")
            elif avg_error_percent < 10:
                print("✅ System is GOOD. Minor errors acceptable for most tasks.")
            elif avg_error_percent < 20:
                print("⚠️ System needs improvement. Consider re-calibrating.")
            else:
                print("❌ System needs re-calibration!")
            
            # Save measurements
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"distance_measurements_{timestamp}.txt"
            with open(filename, 'w') as f:
                f.write("=== DISTANCE MEASUREMENTS ===\n")
                f.write(f"Time: {datetime.now()}\n")
                f.write(f"Workspace: {self.workspace_width_mm}x{self.workspace_height_mm}mm\n\n")
                f.write("Measurements:\n")
                for i, m in enumerate(measurements, 1):
                    f.write(f"{i}. Estimated: {m['estimated']:.0f}mm | Actual: {m['actual']:.0f}mm | Error: {m['error']:.0f}mm ({m['error_percent']:.1f}%)\n")
                f.write(f"\nAverage error: {avg_error:.1f}mm ({avg_error_percent:.1f}%)\n")
            
            print(f"\n✅ Measurements saved to: {filename}")
    
    def run_validation(self):
        """
        Main validation loop
        """
        if not self.is_camera_open:
            if not self.open_camera():
                return
        
        print("\n" + "="*60)
        print("📐 STARTING VALIDATION PROCESS")
        print("="*60)
        print("Steps:")
        print("  1. Place corner markers (IDs 0-3)")
        print("  2. System will setup workspace")
        print("  3. Measure corner distances")
        print("  4. Calculate camera height")
        print("  5. Show verification grid")
        print("  6. Press 'm' to measure chessboard distances")
        print("  7. Press 'q' to quit")
        print("="*60)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Update FPS
            self.frame_count += 1
            if time.time() - self.start_time > 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.start_time = time.time()
            
            if self.is_calibrated:
                frame = self.undistort_frame(frame)
            
            # Detect markers
            self.detect_markers(frame)
            
            # Setup workspace if not done
            if not self.is_workspace_setup:
                if self.setup_workspace(frame):
                    print("\n✅ Workspace setup complete!")
                    self.validate_corner_distances()
                    self.calculate_camera_height()
                    grid_points, pixel_points = self.create_verification_grid(500)
                    frame = self.draw_validation_grid(frame, grid_points, pixel_points)
                    
                    print("\n📊 Place a tape measure along the grid lines")
                    print("   Verify that each 500mm interval matches")
            
            # Draw markers on frame
            for marker_id, data in self.detected_markers.items():
                corners = data['corners'].astype(np.int32)
                center = data['center']
                
                if marker_id in [0, 1, 2, 3]:
                    color = (0, 255, 0)
                elif marker_id in [10, 11]:
                    color = (255, 165, 0)
                elif marker_id in [20, 21]:
                    color = (255, 0, 255)
                elif marker_id in [100, 101, 102]:
                    color = (0, 0, 255)
                else:
                    color = (255, 255, 0)
                
                cv2.polylines(frame, [corners], True, color, 2)
                cv2.circle(frame, center, 5, color, -1)
                
                if self.is_workspace_setup:
                    world_x, world_y = self.get_world_coordinates(center[0], center[1])
                    cv2.putText(frame, f"ID:{marker_id} ({world_x:.0f},{world_y:.0f})", 
                               (center[0]-40, center[1]-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    cv2.putText(frame, f"ID:{marker_id}", 
                               (center[0]-20, center[1]-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Info overlay
            h, w = frame.shape[:2]
            
            cv2.putText(frame, f"FPS: {self.fps}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if self.is_workspace_setup:
                cv2.putText(frame, "✅ Workspace: SET", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Size: {self.workspace_width_mm}x{self.workspace_height_mm}mm", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                if self.camera_height_mm:
                    cv2.putText(frame, f"Camera Height: {self.camera_height_mm/1000:.2f}m", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                cv2.putText(frame, "⏳ Waiting for corner markers...", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, "Place IDs 0,1,2,3 at workspace corners", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, "Controls: q=Quit, m=Measure chessboard, r=Reset", 
                       (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Calibration Validation', frame)
            
            # Show top-down view
            if self.is_workspace_setup:
                warped = cv2.warpPerspective(
                    frame, self.perspective_matrix, 
                    (self.workspace_width_mm, self.workspace_height_mm)
                )
                
                grid_points, pixel_points = self.create_verification_grid(500)
                if grid_points:
                    for world_x, world_y in grid_points:
                        px = int(world_x)
                        py = int(world_y)
                        if px < self.workspace_width_mm and py < self.workspace_height_mm:
                            cv2.circle(warped, (px, py), 3, (0, 255, 255), -1)
                
                display_height = 600
                display_width = int(self.workspace_width_mm * (display_height / self.workspace_height_mm))
                warped_display = cv2.resize(warped, (display_width, display_height))
                cv2.imshow('Top-Down View (mm coordinates)', warped_display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('m'):
                if self.is_workspace_setup:
                    self.measure_chessboard_distance()
                else:
                    print("⚠️ Please wait for workspace setup first")
            elif key == ord('r'):
                self.perspective_matrix = None
                self.is_workspace_setup = False
                self.boundary_points = {}
                self.measured_distances = {}
                self.grid_points = []
                cv2.destroyWindow('Top-Down View (mm coordinates)')
                print("\n🔄 Reset - Place corner markers again")
        
        self.close_camera()
        cv2.destroyAllWindows()
        
        print("\n📐 Validation complete!")
        if self.measured_distances:
            print("   Check the distance measurements above")
        if self.camera_height_mm:
            print(f"   Camera height: {self.camera_height_mm/1000:.2f} meters")

# Main execution
if __name__ == "__main__":
    print("🚀 Camera Calibration Validation System")
    print("="*60)
    
    # Find calibration file
    import glob
    cal_files = glob.glob("F:/swarm_robots/codes/computer_vision/camera_calibration/motorola_calibration_20260702_215248.npz")
    calibration_file = cal_files[0] if cal_files else None
    
    if calibration_file:
        print(f"✅ Using calibration: {calibration_file}")
    else:
        print("⚠️ No calibration file found!")
        print("   Run camera calibration first")
        exit()
    
    # Create validator
    validator = CalibrationValidator(
        camera_id=1,
        calibration_file=calibration_file
    )
    
    # Run validation
    validator.run_validation()