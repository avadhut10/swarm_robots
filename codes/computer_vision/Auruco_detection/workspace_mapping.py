import cv2
import numpy as np
import json
import time
from datetime import datetime
import os

class WorkspaceMapper:
    """
    Map all markers in the workspace and save their positions
    """
    
    def __init__(self, camera_id=1, calibration_file=None):
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
        
        # Workspace
        self.workspace_width_mm = 2800
        self.workspace_height_mm = 2200
        
        # Marker sizes (YOUR MEASUREMENTS)
        self.marker_sizes_mm = {
            0: 47, 1: 47, 2: 47, 3: 47,
            10: 45, 11: 45,
            20: 33.5, 21: 33.5,
            100: 32, 101: 32, 102: 32,
        }
        
        # Marker labels
        self.marker_labels = {
            0: 'B0 (Top-Left)', 1: 'B1 (Top-Right)',
            2: 'B2 (Bottom-Left)', 3: 'B3 (Bottom-Right)',
            10: 'START', 11: 'END',
            20: 'JOB 1', 21: 'JOB 2',
            100: 'ROBOT 1', 101: 'ROBOT 2', 102: 'ROBOT 3'
        }
        
        # Perspective
        self.perspective_matrix = None
        self.boundary_points = {}
        self.is_workspace_setup = False
        
        # Detection
        self.detected_markers = {}
        self.all_positions = {}
        
        # FPS
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        print("🗺️ Workspace Mapper Initialized")
        print(f"   Workspace: {self.workspace_width_mm}x{self.workspace_height_mm}mm")
        print(f"   Calibration: {'✅' if self.is_calibrated else '❌'}")
    
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
        
        self.is_camera_open = True
        print("✅ Camera opened")
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
                
                self.detected_markers[marker_id] = {
                    'id': marker_id,
                    'corners': marker_corners,
                    'center': (center_x, center_y),
                    'size_mm': self.marker_sizes_mm.get(marker_id, 0)
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
    
    def get_all_positions(self):
        """Get positions of all detected markers"""
        positions = {}
        
        for marker_id, data in self.detected_markers.items():
            pixel_x, pixel_y = data['center']
            world_x, world_y = self.get_world_coordinates(pixel_x, pixel_y)
            
            positions[marker_id] = {
                'id': marker_id,
                'label': self.marker_labels.get(marker_id, f'ID{marker_id}'),
                'world_x': world_x,
                'world_y': world_y,
                'size_mm': data['size_mm']
            }
        
        self.all_positions = positions
        return positions
    
    def draw_markers(self, frame, positions):
        """Draw markers on frame"""
        for marker_id, data in positions.items():
            if marker_id not in self.detected_markers:
                continue
            
            corners = self.detected_markers[marker_id]['corners'].astype(np.int32)
            center = (int(data['world_x']), int(data['world_y']))
            
            # Color coding
            if marker_id in [0, 1, 2, 3]:
                color = (0, 255, 0)  # Green
            elif marker_id in [10, 11]:
                color = (255, 165, 0)  # Orange
            elif marker_id in [20, 21]:
                color = (255, 0, 255)  # Magenta
            elif marker_id in [100, 101, 102]:
                color = (0, 0, 255)  # Red
            else:
                color = (255, 255, 0)  # Yellow
            
            cv2.polylines(frame, [corners], True, color, 2)
            
            # Show label and position
            label = data['label']
            cv2.putText(frame, label, (center[0]-30, center[1]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
            
            pos_text = f"({data['world_x']:.0f}, {data['world_y']:.0f})"
            cv2.putText(frame, pos_text, (center[0]-35, center[1]+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    def save_map(self):
        """Save workspace map to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_filename = f"workspace_map_{timestamp}.json"
        map_data = {
            'timestamp': timestamp,
            'workspace_size_mm': [self.workspace_width_mm, self.workspace_height_mm],
            'marker_sizes_mm': self.marker_sizes_mm,
            'positions': {}
        }
        
        for marker_id, pos in self.all_positions.items():
            map_data['positions'][str(marker_id)] = {
                'label': pos['label'],
                'x_mm': float(pos['world_x']),
                'y_mm': float(pos['world_y']),
                'size_mm': float(pos['size_mm'])
            }
        
        with open(json_filename, 'w') as f:
            json.dump(map_data, f, indent=2)
        
        print(f"✅ Map saved to: {json_filename}")
        
        # Save as readable text
        txt_filename = f"workspace_map_{timestamp}.txt"
        with open(txt_filename, 'w') as f:
            f.write("="*60 + "\n")
            f.write("WORKSPACE MAP\n")
            f.write("="*60 + "\n")
            f.write(f"Time: {datetime.now()}\n")
            f.write(f"Workspace: {self.workspace_width_mm}x{self.workspace_height_mm}mm\n")
            f.write("\nMARKER POSITIONS:\n")
            f.write("-"*40 + "\n")
            
            # Group by type
            f.write("\nBOUNDARY MARKERS:\n")
            for id in [0, 1, 2, 3]:
                if id in self.all_positions:
                    pos = self.all_positions[id]
                    f.write(f"  {pos['label']}: ({pos['world_x']:.1f}, {pos['world_y']:.1f})mm\n")
            
            f.write("\nSTART/END:\n")
            for id in [10, 11]:
                if id in self.all_positions:
                    pos = self.all_positions[id]
                    f.write(f"  {pos['label']}: ({pos['world_x']:.1f}, {pos['world_y']:.1f})mm\n")
            
            f.write("\nJOBS:\n")
            for id in [20, 21]:
                if id in self.all_positions:
                    pos = self.all_positions[id]
                    f.write(f"  {pos['label']}: ({pos['world_x']:.1f}, {pos['world_y']:.1f})mm\n")
            
            f.write("\nROBOTS:\n")
            for id in [100, 101, 102]:
                if id in self.all_positions:
                    pos = self.all_positions[id]
                    f.write(f"  {pos['label']}: ({pos['world_x']:.1f}, {pos['world_y']:.1f})mm\n")
        
        print(f"✅ Text map saved to: {txt_filename}")
    
    def run_mapping(self):
        """Main mapping loop"""
        if not self.is_camera_open:
            if not self.open_camera():
                return
        
        print("\n" + "="*60)
        print("🗺️ WORKSPACE MAPPING")
        print("="*60)
        print("Place ALL your markers in the workspace")
        print("The system will detect and map all of them")
        print("\nControls:")
        print("  s - Save current map")
        print("  r - Reset workspace")
        print("  q - Quit")
        print("="*60)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # FPS
            self.frame_count += 1
            if time.time() - self.start_time > 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.start_time = time.time()
            
            if self.is_calibrated:
                frame = self.undistort_frame(frame)
            
            # Setup workspace if not done
            if not self.is_workspace_setup:
                if self.setup_workspace(frame):
                    print("\n✅ Workspace setup complete!")
                    print(f"   Detected workspace: {self.workspace_width_mm}x{self.workspace_height_mm}mm")
            
            # Detect and get positions
            self.detect_markers(frame)
            positions = self.get_all_positions()
            
            # Draw on frame
            self.draw_markers(frame, positions)
            
            # Info overlay
            h, w = frame.shape[:2]
            
            cv2.putText(frame, f"FPS: {self.fps}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            marker_count = len(positions)
            cv2.putText(frame, f"Markers: {marker_count}/11", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if self.is_workspace_setup:
                cv2.putText(frame, "✅ Workspace ready", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "⏳ Place corner markers", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Show missing markers
            all_ids = [0, 1, 2, 3, 10, 11, 20, 21, 100, 101, 102]
            missing = [id for id in all_ids if id not in positions]
            if missing:
                cv2.putText(frame, f"Missing: {missing}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Controls
            cv2.putText(frame, "s=Save  r=Reset  q=Quit", 
                       (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Workspace Mapping', frame)
            
            # Show top-down view
            if self.is_workspace_setup:
                warped = cv2.warpPerspective(
                    frame, self.perspective_matrix, 
                    (self.workspace_width_mm, self.workspace_height_mm)
                )
                
                # Draw marker positions on top-down view
                for marker_id, pos in positions.items():
                    x = int(pos['world_x'])
                    y = int(pos['world_y'])
                    
                    if 0 <= x <= self.workspace_width_mm and 0 <= y <= self.workspace_height_mm:
                        if marker_id in [0, 1, 2, 3]:
                            color = (0, 255, 0)
                            radius = 8
                        elif marker_id in [10, 11]:
                            color = (255, 165, 0)
                            radius = 6
                        elif marker_id in [20, 21]:
                            color = (255, 0, 255)
                            radius = 6
                        elif marker_id in [100, 101, 102]:
                            color = (0, 0, 255)
                            radius = 10
                        else:
                            color = (255, 255, 0)
                            radius = 5
                        
                        cv2.circle(warped, (x, y), radius, color, -1)
                        cv2.putText(warped, str(marker_id), (x-10, y-15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Resize for display
                display_height = 600
                display_width = int(self.workspace_width_mm * (display_height / self.workspace_height_mm))
                warped_display = cv2.resize(warped, (display_width, display_height))
                cv2.imshow('Top-Down Map', warped_display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                if self.is_workspace_setup and len(positions) > 0:
                    self.save_map()
                else:
                    print("⚠️ No markers detected or workspace not setup")
            elif key == ord('r'):
                self.perspective_matrix = None
                self.is_workspace_setup = False
                self.boundary_points = {}
                self.all_positions = {}
                cv2.destroyWindow('Top-Down Map')
                print("🔄 Reset - Place corner markers again")
        
        self.close_camera()
        cv2.destroyAllWindows()
        print("\n🗺️ Mapping complete!")

# Main execution
if __name__ == "__main__":
    print("🚀 Workspace Mapping System")
    print("="*60)
    
    # Find calibration file
    import glob
    cal_files = glob.glob("F:/swarm_robots/codes/computer_vision/camera_calibration/motorola_calibration_20260702_215248.npz")
    calibration_file = cal_files[0] if cal_files else None
    
    if calibration_file:
        print(f"✅ Using calibration: {calibration_file}")
    else:
        print("⚠️ No calibration found - running without")
    
    # Create mapper
    mapper = WorkspaceMapper(
        camera_id=1,
        calibration_file=calibration_file
    )
    
    # Run mapping
    mapper.run_mapping()