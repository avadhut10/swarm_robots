import cv2
import numpy as np
import time
import json
from datetime import datetime
import os

class ArUcoDetectionSystem:
    """
    ArUco detection with YOUR actual workspace (2800 x 2200 mm)
    """
    
    def __init__(self, camera_id=1, calibration_file=None):
        """
        Initialize detection system with your workspace
        """
        self.camera_id = camera_id
        self.cap = None
        self.is_camera_open = False
        
        # Load camera calibration
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.is_calibrated = False
        
        if calibration_file and os.path.exists(calibration_file):
            self.load_calibration(calibration_file)
        
        # ArUco setup
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        except:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5)
        
        self.parameters = cv2.aruco.DetectorParameters()
        
        # ============================================================
        # YOUR ACTUAL WORKSPACE SIZE
        # ============================================================
        self.workspace_width_mm = 2800   # YOUR MEASUREMENT
        self.workspace_height_mm = 2200  # YOUR MEASUREMENT
        
        # ============================================================
        # YOUR ACTUAL MARKER MEASUREMENTS
        # ============================================================
        self.marker_sizes_mm = {
            # Boundary markers (IDs 0-3) - 47x47mm
            0: 47, 1: 47, 2: 47, 3: 47,
            
            # Start/Stop markers (IDs 10-11) - 45x45mm
            10: 45, 11: 45,
            
            # Job markers (IDs 20-21) - 30x37mm (rectangular)
            20: 33.5, 21: 33.5,  # Average: (30+37)/2
            
            # Robot markers (IDs 100-102) - 27x37mm (rectangular)
            100: 32, 101: 32, 102: 32,  # Average: (27+37)/2
        }
        
        # Store actual dimensions for display
        self.marker_actual_dimensions = {
            0: (47, 47), 1: (47, 47), 2: (47, 47), 3: (47, 47),
            10: (45, 45), 11: (45, 45),
            20: (30, 37), 21: (30, 37),
            100: (27, 37), 101: (27, 37), 102: (27, 37)
        }
        
        # Marker types
        self.marker_types = {
            0: 'boundary', 1: 'boundary', 2: 'boundary', 3: 'boundary',
            10: 'start', 11: 'end',
            20: 'job', 21: 'job',
            100: 'robot', 101: 'robot', 102: 'robot'
        }
        
        # Marker labels
        self.marker_labels = {
            0: 'B0', 1: 'B1', 2: 'B2', 3: 'B3',
            10: 'START', 11: 'END',
            20: 'JOB 1', 21: 'JOB 2',
            100: 'ROBOT 1', 101: 'ROBOT 2', 102: 'ROBOT 3'
        }
        
        # Perspective transform
        self.perspective_matrix = None
        self.boundary_points = {}
        self.is_workspace_setup = False
        
        # Tracking data
        self.detected_markers = {}
        self.robot_positions = {}
        self.task_locations = {}
        self.position_history = {}
        self.history_length = 5
        
        # Performance
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        print("🎯 ArUco Detection System Initialized")
        print("="*60)
        print(f"📐 WORKSPACE: {self.workspace_width_mm} x {self.workspace_height_mm} mm")
        print(f"   (This is what the top-down view shows)")
        print("="*60)
        print("📏 MARKER MEASUREMENTS:")
        print(f"   Boundary (IDs 0-3):  47x47 mm")
        print(f"   Start/Stop (IDs 10-11): 45x45 mm")
        print(f"   Jobs (IDs 20-21):    30x37 mm (avg: 33.5mm)")
        print(f"   Robots (IDs 100-102): 27x37 mm (avg: 32mm)")
        print("="*60)
    
    def load_calibration(self, filename):
        """Load camera calibration"""
        try:
            data = np.load(filename)
            self.camera_matrix = data['camera_matrix']
            self.distortion_coeffs = data['distortion_coeffs']
            self.is_calibrated = True
            print(f"✅ Calibration loaded: {filename}")
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
                
                # Calculate center
                center_x = int(np.mean(marker_corners[:, 0]))
                center_y = int(np.mean(marker_corners[:, 1]))
                
                # Calculate size in pixels
                side1 = np.linalg.norm(marker_corners[0] - marker_corners[1])
                side2 = np.linalg.norm(marker_corners[1] - marker_corners[2])
                pixel_size = (side1 + side2) / 2
                
                # Calculate angle
                dx = marker_corners[1][0] - marker_corners[0][0]
                dy = marker_corners[1][1] - marker_corners[0][1]
                angle = np.arctan2(dy, dx) * 180 / np.pi
                
                # Get marker info
                marker_type = self.marker_types.get(marker_id, 'unknown')
                marker_size_mm = self.marker_sizes_mm.get(marker_id, 0)
                marker_label = self.marker_labels.get(marker_id, f"ID{marker_id}")
                actual_dims = self.marker_actual_dimensions.get(marker_id, (0, 0))
                
                # Estimate distance (if calibrated)
                distance_mm = None
                if self.is_calibrated and marker_size_mm > 0 and pixel_size > 0:
                    fx = self.camera_matrix[0, 0]
                    distance_mm = (marker_size_mm * fx) / pixel_size
                
                self.detected_markers[marker_id] = {
                    'id': marker_id,
                    'type': marker_type,
                    'label': marker_label,
                    'corners': marker_corners,
                    'center': (center_x, center_y),
                    'pixel_size': pixel_size,
                    'size_mm': marker_size_mm,
                    'actual_dimensions': actual_dims,
                    'angle': angle,
                    'distance_mm': distance_mm
                }
        
        return self.detected_markers
    
    def setup_workspace(self, frame):
        """
        Setup workspace using boundary markers (IDs 0-3)
        Creates top-down view in your actual workspace size
        """
        self.detect_markers(frame)
        
        boundary_ids = [0, 1, 2, 3]
        boundary_points = []
        
        for marker_id in boundary_ids:
            if marker_id in self.detected_markers:
                center = self.detected_markers[marker_id]['center']
                boundary_points.append((marker_id, center))
                self.boundary_points[marker_id] = center
        
        # If we have all 4 corners, setup perspective
        if len(boundary_points) == 4:
            # Order points: 0=Top-Left, 1=Top-Right, 2=Bottom-Left, 3=Bottom-Right
            ordered_points = []
            for id in [0, 1, 2, 3]:
                if id in self.boundary_points:
                    ordered_points.append(self.boundary_points[id])
            
            if len(ordered_points) == 4:
                # Source points from camera image
                src_points = np.array(ordered_points, dtype=np.float32)
                
                # Destination points in mm (YOUR ACTUAL WORKSPACE)
                dst_points = np.array([
                    [0, 0],                                    # Top-Left
                    [self.workspace_width_mm, 0],              # Top-Right
                    [0, self.workspace_height_mm],             # Bottom-Left
                    [self.workspace_width_mm, self.workspace_height_mm]  # Bottom-Right
                ], dtype=np.float32)
                
                # Calculate perspective transform
                self.perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                self.is_workspace_setup = True
                
                # Create warped top-down view
                warped = cv2.warpPerspective(
                    frame, 
                    self.perspective_matrix, 
                    (self.workspace_width_mm, self.workspace_height_mm)
                )
                
                print(f"\n✅ Workspace setup complete!")
                print(f"   Workspace: {self.workspace_width_mm} x {self.workspace_height_mm} mm")
                print(f"   Top-down view shows real-world coordinates")
                print(f"   Origin (0,0) is Top-Left corner")
                print(f"   X-axis: {self.workspace_width_mm}mm (Left to Right)")
                print(f"   Y-axis: {self.workspace_height_mm}mm (Top to Bottom)")
                
                return warped, True
        
        return frame, False
    
    def get_world_coordinates(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates to world coordinates (mm)
        """
        if self.perspective_matrix is None:
            return pixel_x, pixel_y
        
        pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        world_point = cv2.perspectiveTransform(pixel_point, self.perspective_matrix)
        return world_point[0][0][0], world_point[0][0][1]
    
    def get_all_positions(self):
        """Get positions of all detected markers in world coordinates"""
        positions = {}
        
        for marker_id, data in self.detected_markers.items():
            pixel_x, pixel_y = data['center']
            world_x, world_y = self.get_world_coordinates(pixel_x, pixel_y)
            
            positions[marker_id] = {
                'id': marker_id,
                'type': data['type'],
                'label': data['label'],
                'pixel_x': pixel_x,
                'pixel_y': pixel_y,
                'world_x': world_x,
                'world_y': world_y,
                'angle': data['angle'],
                'size_mm': data['size_mm'],
                'actual_dimensions': data['actual_dimensions'],
                'distance_mm': data['distance_mm']
            }
        
        # Update robot positions
        for marker_id in [100, 101, 102]:
            if marker_id in positions:
                self.robot_positions[marker_id] = positions[marker_id]
                
                if marker_id not in self.position_history:
                    self.position_history[marker_id] = []
                self.position_history[marker_id].append({
                    'x': positions[marker_id]['world_x'],
                    'y': positions[marker_id]['world_y'],
                    'time': time.time()
                })
                if len(self.position_history[marker_id]) > self.history_length:
                    self.position_history[marker_id].pop(0)
        
        # Update task locations
        for marker_id in [10, 11, 20, 21]:
            if marker_id in positions:
                if marker_id == 10:
                    label = "START"
                elif marker_id == 11:
                    label = "END"
                elif marker_id == 20:
                    label = "JOB 1"
                elif marker_id == 21:
                    label = "JOB 2"
                else:
                    label = f"TASK {marker_id}"
                
                self.task_locations[marker_id] = {
                    'id': marker_id,
                    'label': label,
                    'world_x': positions[marker_id]['world_x'],
                    'world_y': positions[marker_id]['world_y']
                }
        
        return positions
    
    def get_smoothed_position(self, robot_id):
        """Get smoothed position for a robot"""
        if robot_id not in self.position_history or len(self.position_history[robot_id]) == 0:
            return None, None
        
        history = self.position_history[robot_id]
        avg_x = sum(p['x'] for p in history) / len(history)
        avg_y = sum(p['y'] for p in history) / len(history)
        
        return avg_x, avg_y
    
    def draw_markers(self, frame, positions):
        """Draw detected markers with information"""
        for marker_id, data in positions.items():
            if marker_id in self.detected_markers:
                corners = self.detected_markers[marker_id]['corners'].astype(np.int32)
            else:
                continue
            
            # Color coding
            marker_type = data['type']
            if marker_type == 'boundary':
                color = (0, 255, 0)  # Green
            elif marker_type == 'robot':
                color = (0, 0, 255)  # Red
            elif marker_type in ['start', 'end']:
                color = (255, 165, 0)  # Orange
            elif marker_type == 'job':
                color = (255, 0, 255)  # Magenta
            else:
                color = (255, 255, 0)  # Yellow
            
            # Draw bounding box
            cv2.polylines(frame, [corners], True, color, 2)
            
            # Draw center
            center = (int(data['pixel_x']), int(data['pixel_y']))
            cv2.circle(frame, center, 5, color, -1)
            
            # Draw label
            label = data['label']
            cv2.putText(frame, label, (center[0]-30, center[1]-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Show world coordinates
            pos_text = f"({data['world_x']:.0f}, {data['world_y']:.0f})mm"
            cv2.putText(frame, pos_text, (center[0]-40, center[1]+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_top_down_view(self, frame):
        """Create and display the top-down view with annotations"""
        if not self.is_workspace_setup or self.perspective_matrix is None:
            return None
        
        # Create warped view
        warped = cv2.warpPerspective(
            frame, 
            self.perspective_matrix, 
            (self.workspace_width_mm, self.workspace_height_mm)
        )
        
        # Add grid lines for better understanding
        # Horizontal lines every 500mm
        for y in range(0, self.workspace_height_mm, 500):
            cv2.line(warped, (0, y), (self.workspace_width_mm, y), (200, 200, 200), 1)
            cv2.putText(warped, f"Y={y}mm", (10, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Vertical lines every 500mm
        for x in range(0, self.workspace_width_mm, 500):
            cv2.line(warped, (x, 0), (x, self.workspace_height_mm), (200, 200, 200), 1)
            cv2.putText(warped, f"X={x}mm", (x+5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw boundary
        cv2.rectangle(warped, (0, 0), 
                     (self.workspace_width_mm, self.workspace_height_mm), 
                     (0, 255, 0), 3)
        
        # Label workspace size
        cv2.putText(warped, f"WORKSPACE: {self.workspace_width_mm}x{self.workspace_height_mm}mm", 
                   (self.workspace_width_mm//2 - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Origin label
        cv2.putText(warped, "ORIGIN (0,0)", (10, self.workspace_height_mm - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw robot positions on top-down view
        for robot_id in [100, 101, 102]:
            if robot_id in self.robot_positions:
                pos = self.robot_positions[robot_id]
                x = int(pos['world_x'])
                y = int(pos['world_y'])
                
                # Only draw if within workspace
                if 0 <= x <= self.workspace_width_mm and 0 <= y <= self.workspace_height_mm:
                    cv2.circle(warped, (x, y), 15, (0, 0, 255), -1)
                    cv2.putText(warped, f"R{robot_id-99}", (x-15, y-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw task locations on top-down view
        for task_id in [10, 11, 20, 21]:
            if task_id in self.task_locations:
                task = self.task_locations[task_id]
                x = int(task['world_x'])
                y = int(task['world_y'])
                
                if 0 <= x <= self.workspace_width_mm and 0 <= y <= self.workspace_height_mm:
                    color = (255, 165, 0) if task_id in [10, 11] else (255, 0, 255)
                    cv2.circle(warped, (x, y), 10, color, -1)
                    cv2.putText(warped, task['label'], (x-20, y-15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Resize for display
        display_height = 700
        display_width = int(self.workspace_width_mm * (display_height / self.workspace_height_mm))
        warped_display = cv2.resize(warped, (display_width, display_height))
        
        return warped_display
    
    def draw_info_overlay(self, frame):
        """Draw information overlay on main frame"""
        h, w = frame.shape[:2]
        y_offset = 30
        
        # FPS
        self.frame_count += 1
        if time.time() - self.start_time > 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.start_time = time.time()
        
        cv2.putText(frame, f"FPS: {self.fps}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        
        # Marker count
        marker_count = len(self.detected_markers)
        cv2.putText(frame, f"Markers: {marker_count}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # Workspace status
        status = "✅" if self.is_workspace_setup else "❌"
        cv2.putText(frame, f"Workspace: {status}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # Workspace size
        cv2.putText(frame, f"Size: {self.workspace_width_mm}x{self.workspace_height_mm}mm", 
                   (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 25
        
        # Calibration status
        cal_status = "✅" if self.is_calibrated else "❌"
        cv2.putText(frame, f"Calibrated: {cal_status}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Robot positions on right side
        if self.is_workspace_setup:
            robot_x = w - 320
            robot_y = 30
            
            cv2.putText(frame, "🤖 ROBOTS:", (robot_x, robot_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            robot_y += 25
            
            for robot_id in [100, 101, 102]:
                if robot_id in self.robot_positions:
                    pos = self.robot_positions[robot_id]
                    smooth_x, smooth_y = self.get_smoothed_position(robot_id)
                    
                    if smooth_x is not None:
                        pos_text = f"R{robot_id-99}: ({smooth_x:.1f}, {smooth_y:.1f})mm"
                    else:
                        pos_text = f"R{robot_id-99}: ({pos['world_x']:.1f}, {pos['world_y']:.1f})mm"
                    
                    cv2.putText(frame, pos_text, (robot_x, robot_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    robot_y += 18
            
            # Task locations
            task_x = w - 320
            task_y = 130
            
            cv2.putText(frame, "📍 TASKS:", (task_x, task_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            task_y += 25
            
            for task_id in [10, 11, 20, 21]:
                if task_id in self.task_locations:
                    task = self.task_locations[task_id]
                    pos_text = f"{task['label']}: ({task['world_x']:.1f}, {task['world_y']:.1f})mm"
                    
                    cv2.putText(frame, pos_text, (task_x, task_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    task_y += 18
    
    def run_detection(self):
        """Main detection loop"""
        if not self.is_camera_open:
            if not self.open_camera():
                return
        
        print("\n" + "="*60)
        print("🎯 ARUCO DETECTION RUNNING")
        print("="*60)
        print(f"📐 WORKSPACE: {self.workspace_width_mm} x {self.workspace_height_mm} mm")
        print("Controls:")
        print("  q - Quit")
        print("  s - Save layout")
        print("  r - Reset workspace")
        print("  Space - Pause")
        print("="*60)
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                if self.is_calibrated:
                    frame = self.undistort_frame(frame)
                
                # Setup workspace if not done
                if not self.is_workspace_setup:
                    warped, success = self.setup_workspace(frame)
                    if success:
                        print("\n📐 Top-down view active!")
                        print(f"   Green box = {self.workspace_width_mm}x{self.workspace_height_mm}mm workspace")
                        print("   Grid lines every 500mm")
                        print("   Origin (0,0) = Top-Left corner")
                
                # Detect and draw markers
                self.detect_markers(frame)
                positions = self.get_all_positions()
                self.draw_markers(frame, positions)
                self.draw_info_overlay(frame)
                
                # Create top-down view
                top_down = self.draw_top_down_view(frame)
                
                # Show main view
                cv2.imshow('ArUco Detection - Main View', frame)
                
                # Show top-down view if available
                if top_down is not None:
                    cv2.imshow('Top-Down View (Real mm positions)', top_down)
                else:
                    # Show message if workspace not setup
                    info_frame = np.ones((200, 600, 3), dtype=np.uint8) * 50
                    cv2.putText(info_frame, "Waiting for workspace setup...", (50, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(info_frame, "Place boundary markers (IDs 0-3)", (50, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.imshow('Top-Down View (Real mm positions)', info_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_layout()
            elif key == ord('r'):
                self.perspective_matrix = None
                self.is_workspace_setup = False
                self.boundary_points = {}
                print("\n🔄 Workspace reset")
                cv2.destroyWindow('Top-Down View (Real mm positions)')
            elif key == 32:
                paused = not paused
                print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
        
        self.close_camera()
        cv2.destroyAllWindows()
        print("\n✅ Detection stopped")
    
    def save_layout(self):
        """Save current layout"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"swarm_layout_{timestamp}.json"
        
        layout_data = {
            'timestamp': timestamp,
            'workspace_size_mm': [self.workspace_width_mm, self.workspace_height_mm],
            'marker_measurements': self.marker_actual_dimensions,
            'task_locations': self.task_locations,
            'robot_positions': {
                str(id): {'x': pos['world_x'], 'y': pos['world_y'], 'angle': pos.get('angle', 0)}
                for id, pos in self.robot_positions.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(layout_data, f, indent=2)
        
        print(f"✅ Layout saved to: {filename}")
        
        # Also save as readable text
        txt_filename = f"swarm_layout_{timestamp}.txt"
        with open(txt_filename, 'w') as f:
            f.write("=== SWARM LAYOUT ===\n")
            f.write(f"Time: {datetime.now()}\n")
            f.write(f"Workspace: {self.workspace_width_mm}x{self.workspace_height_mm}mm\n\n")
            
            f.write("TASK LOCATIONS:\n")
            for id, task in self.task_locations.items():
                f.write(f"  {task['label']} (ID {id}): ({task['world_x']:.2f}, {task['world_y']:.2f})mm\n")
            
            f.write("\nROBOT POSITIONS:\n")
            for id, pos in self.robot_positions.items():
                f.write(f"  Robot {id-99} (ID {id}): ({pos['world_x']:.2f}, {pos['world_y']:.2f})mm\n")
        
        print(f"✅ Text layout saved to: {txt_filename}")

# Main execution
if __name__ == "__main__":
    print("🚀 ArUco Detection with YOUR Workspace")
    print("="*60)
    
    # Find calibration file
    import glob
    cal_files = glob.glob("F:/swarm_robots/codes/computer_vision/camera_calibration/motorola_calibration_20260702_215248.npz")
    calibration_file = cal_files[0] if cal_files else None
    
    if calibration_file:
        print(f"✅ Found calibration: {calibration_file}")
    else:
        print("⚠️ No calibration file found")
        print("   Distance measurements won't be accurate")
    
    # Create detector with YOUR workspace size
    detector = ArUcoDetectionSystem(
        camera_id=1,
        calibration_file=calibration_file
    )
    
    # Run
    detector.run_detection()