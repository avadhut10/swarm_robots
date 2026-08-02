import cv2
import numpy as np
import json
import time
import math
from datetime import datetime
import os

class RealTimeWorkspaceDetector:
    """
    Real-time workspace detector with graphical boundary
    Detects ALL markers and shows positions simultaneously
    """
    
    def __init__(self, camera_id=1, calibration_file=None):
        """
        Initialize the real-time detector
        
        Args:
            camera_id: Camera device ID (1 for Motorola Smart Connect)
            calibration_file: Path to calibration .npz file
        """
        self.camera_id = camera_id
        self.cap = None
        self.is_camera_open = False
        
        # Load calibration
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
        
        # YOUR MEASUREMENTS
        self.marker_sizes_mm = {
            0: 47, 1: 47, 2: 47, 3: 47,
            10: 45, 11: 45,
            20: 33.5, 21: 33.5,
            100: 32, 101: 32, 102: 32,
        }
        
        # Workspace
        self.workspace_width_mm = 2800
        self.workspace_height_mm = 2200
        
        # Detection data
        self.detected_markers = {}
        self.corner_markers = {}  # IDs 0-3
        self.start_marker = None  # ID 10
        self.end_marker = None    # ID 11
        self.task_markers = {}    # IDs 20-21
        self.robot_markers = {}   # IDs 100-102
        
        # Boundary
        self.boundary_points = {}
        self.is_boundary_set = False
        
        # Perspective transform
        self.perspective_matrix = None
        
        # FPS
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Status messages
        self.status_message = "Place corner markers (IDs 0-3)"
        self.status_color = (0, 0, 255)  # Red
        
        print("="*60)
        print("🎯 REAL-TIME WORKSPACE DETECTOR")
        print("="*60)
        print("This system detects ALL markers and shows them in real-time")
        print("")
        print("MARKER TYPES:")
        print("  🟩 IDs 0-3    = BOUNDARY (Green squares)")
        print("  ⭐ ID 10      = START (Lime star)")
        print("  ⭐ ID 11      = END (Red star)")
        print("  🟧 IDs 20-21  = JOBS (Orange squares)")
        print("  🔵 IDs 100-102 = ROBOTS (Blue circles)")
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
        """Open camera"""
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
        if self.cap is not None:
            self.cap.release()
            self.is_camera_open = False
    
    def undistort_frame(self, frame):
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
        """Detect ALL ArUco markers"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        try:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.parameters
            )
        except:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict
            )
        
        # Clear previous detections
        self.detected_markers = {}
        self.corner_markers = {}
        self.task_markers = {}
        self.robot_markers = {}
        self.start_marker = None
        self.end_marker = None
        
        if ids is not None and len(ids) > 0:
            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                marker_corners = corners[i][0]
                
                center_x = int(np.mean(marker_corners[:, 0]))
                center_y = int(np.mean(marker_corners[:, 1]))
                
                # Calculate angle
                dx = marker_corners[1][0] - marker_corners[0][0]
                dy = marker_corners[1][1] - marker_corners[0][1]
                angle = np.arctan2(dy, dx) * 180 / np.pi
                
                marker_data = {
                    'id': marker_id,
                    'corners': marker_corners,
                    'center': (center_x, center_y),
                    'angle': angle,
                    'size_mm': self.marker_sizes_mm.get(marker_id, 0)
                }
                
                self.detected_markers[marker_id] = marker_data
                
                # Classify markers
                if marker_id in [0, 1, 2, 3]:
                    self.corner_markers[marker_id] = marker_data
                elif marker_id == 10:
                    self.start_marker = marker_data
                elif marker_id == 11:
                    self.end_marker = marker_data
                elif marker_id in [20, 21]:
                    self.task_markers[marker_id] = marker_data
                elif marker_id in [100, 101, 102]:
                    self.robot_markers[marker_id] = marker_data
        
        return self.detected_markers
    
    def setup_boundary(self, frame):
        """Setup boundary using corner markers (IDs 0-3)"""
        if len(self.corner_markers) == 4:
            # Order points: 0=TL, 1=TR, 2=BL, 3=BR
            ordered_points = []
            for id in [0, 1, 2, 3]:
                if id in self.corner_markers:
                    ordered_points.append(self.corner_markers[id]['center'])
                    self.boundary_points[id] = self.corner_markers[id]['center']
            
            if len(ordered_points) == 4:
                self.is_boundary_set = True
                self.status_message = "✅ BOUNDARY SET! Place other markers"
                self.status_color = (0, 255, 0)
                return True
        
        self.is_boundary_set = False
        if len(self.corner_markers) < 4:
            self.status_message = f"⏳ Boundary: {len(self.corner_markers)}/4 corners detected"
            self.status_color = (0, 165, 255)  # Orange
        return False
    
    def draw_boundary(self, frame):
        """Draw the workspace boundary on frame"""
        if not self.is_boundary_set:
            return frame
        
        # Get corner points in order
        points = []
        for id in [0, 1, 2, 3]:
            if id in self.boundary_points:
                points.append(self.boundary_points[id])
        
        if len(points) == 4:
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Draw filled boundary with transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
            
            # Draw boundary lines
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
            
            # Draw corner markers
            for id, point in self.boundary_points.items():
                cv2.circle(frame, point, 10, (0, 255, 0), -1)
                cv2.putText(frame, f"ID{id}", (point[0]-15, point[1]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add corner labels
            labels = ['TL', 'TR', 'BL', 'BR']
            for i, (id, point) in enumerate(self.boundary_points.items()):
                cv2.putText(frame, labels[i], (point[0]-15, point[1]+25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Add workspace dimensions
            if len(points) == 4:
                # Top edge
                mid_x = (points[0][0] + points[1][0]) // 2
                mid_y = (points[0][1] + points[1][1]) // 2 - 30
                cv2.putText(frame, f"{self.workspace_width_mm}mm", 
                           (mid_x-30, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Left edge
                mid_x = (points[0][0] + points[2][0]) // 2 - 60
                mid_y = (points[0][1] + points[2][1]) // 2
                cv2.putText(frame, f"{self.workspace_height_mm}mm", 
                           (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def draw_markers(self, frame):
        """Draw all detected markers with labels"""
        # Draw START marker (ID 10)
        if self.start_marker:
            center = self.start_marker['center']
            corners = self.start_marker['corners'].astype(np.int32)
            
            # Draw star symbol
            cv2.polylines(frame, [corners], True, (0, 255, 255), 3)
            cv2.circle(frame, center, 8, (0, 255, 255), -1)
            cv2.putText(frame, "⭐ START", (center[0]-30, center[1]-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"({center[0]}, {center[1]})", 
                       (center[0]-25, center[1]+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw END marker (ID 11)
        if self.end_marker:
            center = self.end_marker['center']
            corners = self.end_marker['corners'].astype(np.int32)
            
            cv2.polylines(frame, [corners], True, (0, 0, 255), 3)
            cv2.circle(frame, center, 8, (0, 0, 255), -1)
            cv2.putText(frame, "⭐ END", (center[0]-20, center[1]-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"({center[0]}, {center[1]})", 
                       (center[0]-25, center[1]+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw TASK markers (IDs 20-21)
        for marker_id, data in self.task_markers.items():
            center = data['center']
            corners = data['corners'].astype(np.int32)
            
            color = (255, 165, 0)  # Orange
            label = "JOB 1" if marker_id == 20 else "JOB 2"
            
            cv2.polylines(frame, [corners], True, color, 3)
            cv2.circle(frame, center, 8, color, -1)
            cv2.putText(frame, f"📦 {label}", (center[0]-25, center[1]-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"ID{marker_id}", (center[0]-15, center[1]+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw ROBOT markers (IDs 100-102)
        for marker_id, data in self.robot_markers.items():
            center = data['center']
            corners = data['corners'].astype(np.int32)
            
            color = (255, 0, 0)  # Blue
            robot_num = marker_id - 99
            
            cv2.polylines(frame, [corners], True, color, 3)
            cv2.circle(frame, center, 10, color, -1)
            cv2.putText(frame, f"🤖 ROBOT {robot_num}", (center[0]-30, center[1]-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"ID{marker_id}", (center[0]-15, center[1]+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def draw_info_overlay(self, frame):
        """Draw information overlay"""
        h, w = frame.shape[:2]
        y_offset = 30
        
        # FPS
        self.frame_count += 1
        if time.time() - self.start_time > 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.start_time = time.time()
        
        # Status bar at top
        cv2.rectangle(frame, (0, 0), (w, 35), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 35), self.status_color, 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Marker count
        total = len(self.detected_markers)
        cv2.putText(frame, f"Markers: {total}", (100, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Boundary status
        status = "SET ✅" if self.is_boundary_set else "WAITING ⏳"
        cv2.putText(frame, f"Boundary: {status}", (210, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 255, 0) if self.is_boundary_set else (0, 165, 255), 2)
        
        # Status message
        cv2.putText(frame, self.status_message, (w//2 - 150, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.status_color, 2)
        
        # Controls at bottom
        cv2.rectangle(frame, (0, h-25), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "s=Save Layout  r=Reset Boundary  q=Quit", 
                   (w//2 - 150, h-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Legend on right side
        legend_x = w - 200
        legend_y = 60
        
        cv2.rectangle(frame, (legend_x-10, legend_y-10), 
                     (w-10, legend_y + 190), (0, 0, 0), -1)
        cv2.rectangle(frame, (legend_x-10, legend_y-10), 
                     (w-10, legend_y + 190), (255, 255, 255), 1)
        
        cv2.putText(frame, "LEGEND:", (legend_x, legend_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        items = [
            ("🟩 IDs 0-3", "Boundary"),
            ("⭐ ID 10", "START"),
            ("⭐ ID 11", "END"),
            ("🟧 IDs 20-21", "JOBS"),
            ("🔵 IDs 100-102", "ROBOTS")
        ]
        
        for i, (symbol, label) in enumerate(items):
            y = legend_y + 40 + i * 30
            cv2.putText(frame, f"{symbol} = {label}", (legend_x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def save_layout(self):
        """Save current layout to file"""
        if not self.is_boundary_set:
            print("⚠️ Boundary not set! Place corner markers first.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"workspace_map_{timestamp}.json"
        
        # Collect positions
        positions = {}
        
        # Corner markers
        for marker_id, data in self.corner_markers.items():
            center = data['center']
            positions[str(marker_id)] = {
                'label': f'B{marker_id}',
                'x_mm': center[0],
                'y_mm': center[1],
                'size_mm': data['size_mm']
            }
        
        # Start marker
        if self.start_marker:
            center = self.start_marker['center']
            positions['10'] = {
                'label': 'START',
                'x_mm': center[0],
                'y_mm': center[1],
                'size_mm': self.start_marker['size_mm']
            }
        
        # End marker
        if self.end_marker:
            center = self.end_marker['center']
            positions['11'] = {
                'label': 'END',
                'x_mm': center[0],
                'y_mm': center[1],
                'size_mm': self.end_marker['size_mm']
            }
        
        # Task markers
        for marker_id, data in self.task_markers.items():
            center = data['center']
            label = 'JOB 1' if marker_id == 20 else 'JOB 2'
            positions[str(marker_id)] = {
                'label': label,
                'x_mm': center[0],
                'y_mm': center[1],
                'size_mm': data['size_mm']
            }
        
        # Robot markers
        for marker_id, data in self.robot_markers.items():
            center = data['center']
            robot_num = marker_id - 99
            positions[str(marker_id)] = {
                'label': f'ROBOT {robot_num}',
                'x_mm': center[0],
                'y_mm': center[1],
                'size_mm': data['size_mm']
            }
        
        # Save to JSON
        map_data = {
            'timestamp': timestamp,
            'workspace_size_mm': [self.workspace_width_mm, self.workspace_height_mm],
            'positions': positions,
            'marker_sizes_mm': self.marker_sizes_mm,
            'boundary_set': self.is_boundary_set
        }
        
        with open(filename, 'w') as f:
            json.dump(map_data, f, indent=2)
        
        print(f"\n✅ Layout saved to: {filename}")
        print(f"   Total markers: {len(positions)}")
        print(f"   Boundary: {'✅ Set' if self.is_boundary_set else '❌ Not set'}")
        
        # Print summary
        print("\n📊 DETECTED MARKERS:")
        for marker_id, data in self.detected_markers.items():
            center = data['center']
            marker_type = "Unknown"
            if marker_id in [0, 1, 2, 3]:
                marker_type = "Boundary"
            elif marker_id == 10:
                marker_type = "START"
            elif marker_id == 11:
                marker_type = "END"
            elif marker_id in [20, 21]:
                marker_type = "JOB"
            elif marker_id in [100, 101, 102]:
                marker_type = "ROBOT"
            
            print(f"   ID {marker_id} ({marker_type}): ({center[0]}, {center[1]})")
    
    def run(self):
        """Main detection loop"""
        if not self.is_camera_open:
            if not self.open_camera():
                return
        
        print("\n" + "="*60)
        print("🔴 REAL-TIME DETECTOR RUNNING")
        print("="*60)
        print("Place your markers in the workspace")
        print("The boundary will form when all 4 corner markers are detected")
        print("Press 's' to save the layout")
        print("Press 'r' to reset boundary")
        print("Press 'q' to quit")
        print("="*60)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            if self.is_calibrated:
                frame = self.undistort_frame(frame)
            
            # Detect markers
            self.detect_markers(frame)
            
            # Setup boundary
            self.setup_boundary(frame)
            
            # Draw everything
            frame = self.draw_boundary(frame)
            frame = self.draw_markers(frame)
            frame = self.draw_info_overlay(frame)
            
            # Show frame
            cv2.imshow('Real-Time Workspace Detector', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_layout()
            elif key == ord('r'):
                self.boundary_points = {}
                self.is_boundary_set = False
                self.status_message = "🔄 Boundary reset - Place corner markers again"
                self.status_color = (0, 165, 255)
                print("\n🔄 Boundary reset")
        
        self.close_camera()
        cv2.destroyAllWindows()
        print("\n✅ Detector stopped")


# Main execution
if __name__ == "__main__":
    print("🚀 REAL-TIME WORKSPACE DETECTOR")
    print("="*60)
    
    # Find calibration file
    import glob
    cal_files = glob.glob("motorola_calibration_*.npz")
    calibration_file = cal_files[0] if cal_files else None
    
    if calibration_file:
        print(f"✅ Using calibration: {calibration_file}")
    else:
        print("⚠️ No calibration found - running without")
    
    # Create detector
    detector = RealTimeWorkspaceDetector(
        camera_id=1,
        calibration_file=calibration_file
    )
    
    # Run
    detector.run()