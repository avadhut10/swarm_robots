import cv2
import numpy as np
import time
import math
from datetime import datetime

class ArUcoDetector:
    def __init__(self, camera_id=0, marker_size_mm=None):
        """
        Initialize the ArUco detector
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
            marker_size_mm: Dictionary of marker sizes in mm for each ID
        """
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        # Set camera properties (optional - adjust based on your camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # ArUco dictionary setup
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        except:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5)
        
        # Parameters for detection
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 10
        
        # Marker sizes in mm (from your printed markers)
        # UPDATE THESE VALUES AFTER PRINTING AND MEASURING!
        self.marker_sizes_mm = marker_size_mm or {
            0: 80, 1: 80, 2: 80, 3: 80,      # Boundary
            10: 60, 11: 60,                   # Start/Stop
            20: 50, 21: 50,                   # Job
            100: 40, 101: 40, 102: 40         # Robots
        }
        
        # Camera matrix and distortion coefficients (for accurate distance)
        # You need to calibrate your camera to get these values
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Store detected markers
        self.detected_markers = {}
        self.boundary_points = {}
        self.robot_positions = {}
        self.task_locations = {}
        
        # Perspective transform matrix (for top-down view)
        self.perspective_matrix = None
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
    
    def load_camera_calibration(self, camera_matrix_file="camera_calibration.npz"):
        """Load camera calibration parameters from file"""
        try:
            data = np.load(camera_matrix_file)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            print("✅ Camera calibration loaded successfully")
            return True
        except:
            print("⚠️ No camera calibration found. Using default values.")
            # Use default values (will be less accurate)
            self.camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
            self.dist_coeffs = np.zeros((4, 1))
            return False
    
    def calibrate_camera(self, chessboard_path="chessboard_calibration.pdf", num_frames=20):
        """
        Simple camera calibration using a chessboard pattern
        Print the chessboard and hold it in front of the camera
        """
        print("📐 Starting camera calibration...")
        print("   Hold the chessboard pattern in front of the camera")
        print("   Move it to different positions and angles")
        print(f"   Capturing {num_frames} frames...")
        
        # Chessboard parameters
        chessboard_size = (9, 6)
        square_size = 25  # mm (adjust based on your printed chessboard)
        
        # Prepare object points
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp = objp * square_size
        
        obj_points = []
        img_points = []
        
        captured = 0
        while captured < num_frames:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            
            if ret_corners:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Draw corners on frame
                cv2.drawChessboardCorners(frame, chessboard_size, corners_refined, ret_corners)
                
                obj_points.append(objp)
                img_points.append(corners_refined)
                captured += 1
                
                print(f"   Calibration progress: {captured}/{num_frames}")
            
            # Show frame
            cv2.imshow('Calibration - Press ESC to skip', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
        
        cv2.destroyWindow('Calibration')
        
        if len(obj_points) > 0:
            # Calibrate camera
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, gray.shape[::-1], None, None
            )
            
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            
            # Save calibration
            np.savez('camera_calibration.npz', 
                    camera_matrix=camera_matrix, 
                    dist_coeffs=dist_coeffs)
            
            print("✅ Camera calibration completed!")
            print(f"   Camera Matrix:\n{camera_matrix}")
            print(f"   Distortion Coefficients: {dist_coeffs.ravel()}")
            return True
        else:
            print("⚠️ Calibration failed! Using default values.")
            self.camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
            self.dist_coeffs = np.zeros((4, 1))
            return False
    
    def detect_markers(self, frame):
        """Detect ArUco markers in the frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        try:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.parameters
            )
        except:
            # Try older API
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict
            )
        
        # Store detected markers
        self.detected_markers = {}
        
        if ids is not None and len(ids) > 0:
            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                marker_corners = corners[i][0]  # 4 corner points
                
                # Calculate center
                center_x = int(np.mean(marker_corners[:, 0]))
                center_y = int(np.mean(marker_corners[:, 1]))
                
                # Calculate marker size in pixels
                side1 = np.linalg.norm(marker_corners[0] - marker_corners[1])
                side2 = np.linalg.norm(marker_corners[1] - marker_corners[2])
                pixel_size = (side1 + side2) / 2
                
                # Calculate distance if camera is calibrated
                distance_mm = None
                if self.camera_matrix is not None and marker_id in self.marker_sizes_mm:
                    # Get actual marker size
                    actual_size_mm = self.marker_sizes_mm[marker_id]
                    
                    # Estimate distance (simplified)
                    # For more accuracy, use solvePnP
                    if pixel_size > 0:
                        # Focal length approximation from camera matrix
                        fx = self.camera_matrix[0, 0]
                        distance_mm = (actual_size_mm * fx) / pixel_size
                
                # Store marker data
                self.detected_markers[marker_id] = {
                    'id': marker_id,
                    'corners': marker_corners,
                    'center': (center_x, center_y),
                    'pixel_size': pixel_size,
                    'size_mm': self.marker_sizes_mm.get(marker_id, 0),
                    'distance_mm': distance_mm,
                    'angle': self.calculate_marker_angle(marker_corners)
                }
        
        return self.detected_markers
    
    def calculate_marker_angle(self, corners):
        """Calculate the rotation angle of a marker"""
        # Vector from top-left to top-right
        top_left = corners[0]
        top_right = corners[1]
        
        dx = top_right[0] - top_left[0]
        dy = top_right[1] - top_left[1]
        
        angle = np.arctan2(dy, dx) * 180 / np.pi
        return angle
    
    def setup_boundary(self, frame):
        """Detect boundary markers and setup perspective transform"""
        # Detect all markers
        self.detect_markers(frame)
        
        # Get boundary markers (IDs 0-3)
        boundary_ids = [0, 1, 2, 3]
        boundary_points = []
        
        for marker_id in boundary_ids:
            if marker_id in self.detected_markers:
                center = self.detected_markers[marker_id]['center']
                boundary_points.append((marker_id, center))
                self.boundary_points[marker_id] = center
        
        # If we have all 4 boundary markers, setup perspective transform
        if len(boundary_points) == 4:
            # Order points: Top-Left, Top-Right, Bottom-Right, Bottom-Left
            # Based on ID: 0=TL, 1=TR, 2=BL, 3=BR
            ordered_points = []
            for id in [0, 1, 2, 3]:
                if id in self.boundary_points:
                    ordered_points.append(self.boundary_points[id])
            
            if len(ordered_points) == 4:
                # Define output size (in mm)
                output_width = 2000  # 2 meters
                output_height = 1500  # 1.5 meters
                
                src_points = np.array(ordered_points, dtype=np.float32)
                dst_points = np.array([
                    [0, 0],
                    [output_width, 0],
                    [0, output_height],
                    [output_width, output_height]
                ], dtype=np.float32)
                
                # Calculate perspective transform
                self.perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                
                # Create a warped view for visualization
                warped = cv2.warpPerspective(frame, self.perspective_matrix, 
                                            (output_width, output_height))
                
                return warped, True
        
        return frame, False
    
    def get_robot_positions(self):
        """Get positions of all detected robots (IDs 100-102)"""
        robot_ids = [100, 101, 102]
        positions = {}
        
        for robot_id in robot_ids:
            if robot_id in self.detected_markers:
                marker = self.detected_markers[robot_id]
                
                # Get pixel coordinates
                pixel_x, pixel_y = marker['center']
                
                # Convert to world coordinates if perspective matrix exists
                world_x, world_y = pixel_x, pixel_y
                if self.perspective_matrix is not None:
                    # Transform pixel to world coordinates
                    pixel_point = np.array([[pixel_x, pixel_y]], dtype=np.float32)
                    pixel_point = np.array([pixel_point])
                    world_point = cv2.perspectiveTransform(pixel_point, self.perspective_matrix)
                    world_x, world_y = world_point[0][0]
                
                positions[robot_id] = {
                    'id': robot_id,
                    'pixel_x': pixel_x,
                    'pixel_y': pixel_y,
                    'world_x': world_x,
                    'world_y': world_y,
                    'distance_mm': marker['distance_mm'],
                    'angle': marker['angle']
                }
        
        return positions
    
    def get_task_locations(self):
        """Get positions of task markers (IDs 10, 11, 20, 21)"""
        task_ids = [10, 11, 20, 21]
        locations = {}
        
        for task_id in task_ids:
            if task_id in self.detected_markers:
                marker = self.detected_markers[task_id]
                
                # Get pixel coordinates
                pixel_x, pixel_y = marker['center']
                
                # Convert to world coordinates if perspective matrix exists
                world_x, world_y = pixel_x, pixel_y
                if self.perspective_matrix is not None:
                    pixel_point = np.array([[pixel_x, pixel_y]], dtype=np.float32)
                    pixel_point = np.array([pixel_point])
                    world_point = cv2.perspectiveTransform(pixel_point, self.perspective_matrix)
                    world_x, world_y = world_point[0][0]
                
                # Determine type
                if task_id in [10, 11]:
                    marker_type = "start_stop"
                    label = "Start" if task_id == 10 else "End"
                else:
                    marker_type = "job"
                    label = f"Job {task_id-19}"
                
                locations[task_id] = {
                    'id': task_id,
                    'type': marker_type,
                    'label': label,
                    'pixel_x': pixel_x,
                    'pixel_y': pixel_y,
                    'world_x': world_x,
                    'world_y': world_y
                }
        
        return locations
    
    def draw_markers(self, frame, show_info=True):
        """Draw detected markers on the frame with information"""
        for marker_id, data in self.detected_markers.items():
            corners = data['corners'].astype(np.int32)
            center = data['center']
            
            # Determine color based on type
            if marker_id in [0, 1, 2, 3]:
                color = (0, 255, 0)  # Green for boundary
                label = f"Boundary {marker_id}"
            elif marker_id in [10, 11]:
                color = (255, 165, 0)  # Orange for start/stop
                label = "Start" if marker_id == 10 else "End"
            elif marker_id in [20, 21]:
                color = (255, 0, 255)  # Magenta for jobs
                label = f"Job {marker_id-19}"
            elif marker_id in [100, 101, 102]:
                color = (0, 0, 255)  # Red for robots
                label = f"Robot {marker_id-99}"
            else:
                color = (255, 255, 0)  # Cyan for others
                label = f"ID: {marker_id}"
            
            # Draw bounding box
            cv2.polylines(frame, [corners], True, color, 2)
            
            # Draw center
            cv2.circle(frame, center, 5, color, -1)
            
            # Draw ID and info
            if show_info:
                info_text = f"{label}"
                cv2.putText(frame, info_text, (center[0] - 30, center[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Show size and distance
                size_mm = data['size_mm']
                if data['distance_mm']:
                    dist_text = f"{size_mm}mm, {data['distance_mm']/1000:.2f}m"
                else:
                    dist_text = f"{size_mm}mm"
                cv2.putText(frame, dist_text, (center[0] - 30, center[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run_detection(self):
        """Main detection loop"""
        print("\n🔍 Starting ArUco marker detection...")
        print("   Press 'q' to quit")
        print("   Press 'c' to calibrate camera")
        print("   Press 'r' to reset perspective")
        print("   Press 's' to save current layout")
        
        # Try to load camera calibration
        self.load_camera_calibration()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Detect markers
            markers = self.detect_markers(frame)
            
            # Setup boundary and perspective
            warped_frame, has_boundary = self.setup_boundary(frame)
            
            # Get robot positions
            if has_boundary:
                robot_positions = self.get_robot_positions()
                task_locations = self.get_task_locations()
                
                # Draw robot positions
                for robot_id, pos in robot_positions.items():
                    cv2.putText(frame, f"Robot {robot_id-99}: ({pos['world_x']:.0f}, {pos['world_y']:.0f})mm",
                               (10, 30 + (robot_id-100) * 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Draw task locations
                for task_id, loc in task_locations.items():
                    cv2.putText(frame, f"{loc['label']}: ({loc['world_x']:.0f}, {loc['world_y']:.0f})mm",
                               (10, 120 + (task_id-10) * 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw markers on frame
            self.draw_markers(frame)
            
            # Show FPS
            self.frame_count += 1
            if time.time() - self.start_time > 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.start_time = time.time()
            
            cv2.putText(frame, f"FPS: {self.fps}", (frame.shape[1] - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show marker count
            cv2.putText(frame, f"Markers: {len(markers)}", (frame.shape[1] - 120, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show perspective status
            if has_boundary:
                cv2.putText(frame, "Boundary: SET", (frame.shape[1] - 150, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Boundary: NOT SET", (frame.shape[1] - 160, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow('ArUco Marker Detection', frame)
            
            # Show warped view if perspective is set
            if has_boundary and warped_frame is not None:
                # Resize warped for display
                display_warped = cv2.resize(warped_frame, (800, 600))
                cv2.imshow('Top-Down View', display_warped)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.calibrate_camera()
            elif key == ord('r'):
                self.perspective_matrix = None
                self.boundary_points = {}
                print("🔄 Perspective reset")
            elif key == ord('s'):
                self.save_layout()
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
    
    def save_layout(self):
        """Save current layout to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"layout_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("=== SWARM ROBOT LAYOUT ===\n")
            f.write(f"Time: {datetime.now()}\n\n")
            
            f.write("BOUNDARY MARKERS:\n")
            for id, pos in self.boundary_points.items():
                f.write(f"  ID {id}: ({pos[0]:.2f}, {pos[1]:.2f}) pixels\n")
            
            f.write("\nTASK LOCATIONS:\n")
            locations = self.get_task_locations()
            for id, loc in locations.items():
                f.write(f"  {loc['label']} (ID {id}): ")
                f.write(f"({loc['world_x']:.2f}, {loc['world_y']:.2f}) mm\n")
            
            f.write("\nROBOT POSITIONS:\n")
            robots = self.get_robot_positions()
            for id, pos in robots.items():
                f.write(f"  Robot {id-99} (ID {id}): ")
                f.write(f"({pos['world_x']:.2f}, {pos['world_y']:.2f}) mm\n")
                if pos['distance_mm']:
                    f.write(f"    Distance: {pos['distance_mm']/1000:.3f} m\n")
        
        print(f"✅ Layout saved to: {filename}")

# Main execution
if __name__ == "__main__":
    print("🚀 ArUco Marker Detection System for Swarm Robots")
    print("="*50)
    
    # Define marker sizes (UPDATE AFTER PRINTING AND MEASURING!)
    MARKER_SIZES_MM = {
        0: 80, 1: 80, 2: 80, 3: 80,      # Boundary markers
        10: 60, 11: 60,                   # Start/Stop markers
        20: 50, 21: 50,                   # Job markers
        100: 40, 101: 40, 102: 40         # Robot markers
    }
    
    # Create detector
    try:
        detector = ArUcoDetector(camera_id=0, marker_size_mm=MARKER_SIZES_MM)
        detector.run_detection()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure your camera is connected and accessible.")