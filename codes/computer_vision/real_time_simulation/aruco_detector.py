import cv2
import cv2.aruco as aruco
import numpy as np
import threading
import time

class ArUcoDetector:
    def __init__(self, camera_id=1, width=2100, height=2850):
        self.camera_id = camera_id
        self.workspace_width = width   # in mm
        self.workspace_height = height # in mm
        
        # Thread control
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Dictionary configuration
        try:
            self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
            self.parameters = aruco.DetectorParameters()
        except AttributeError:
            self.dictionary = aruco.Dictionary_get(aruco.DICT_5X5_1000)
            self.parameters = aruco.DetectorParameters_create()

        # Shared detected positions
        self.robot_positions = {}   # ID (100,101,102) -> (x_mm, y_mm)
        self.task_positions = {}    # ID (20,21) -> (x_mm, y_mm)
        self.start_location = None  # (x_mm, y_mm)
        self.end_location = None    # (x_mm, y_mm)
        self.corner_markers = {}    # ID (0,1,2,3) -> (px, py)
        
        # Perspective transform matrix
        self.perspective_matrix = None
        self.last_debug_time = 0

    def start_detection(self):
        """Starts the background camera thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.thread.start()
            print(f"🚀 [ArUcoDetector] Thread started. Capturing from Camera ID: {self.camera_id}")

    def stop_detection(self):
        """Stops the background camera thread."""
        self.running = False
        if self.thread:
            self.thread.join()
            print("⏹️ [ArUcoDetector] Thread stopped.")

    def _detection_loop(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"❌ [ArUcoDetector] Error: Could not open Camera ID {self.camera_id}. Trying default Camera 0...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ [ArUcoDetector] Error: No camera source found.")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue
                
            self.detect_markers(frame)
            time.sleep(0.01)
            
        cap.release()

    def detect_markers(self, frame):
        """
        Detects markers in a single frame. 
        CRITICAL FIX: Clears all positions first to enforce real-time, zero-fallback accuracy.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        try:
            corners, ids, rejected = aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        except Exception as e:
            corners, ids, rejected = aruco.detectMarkers(gray, self.dictionary)

        with self.lock:
            # 1. CRITICAL: CLEAR ALL POSITIONS FIRST EVERY FRAME
            self.robot_positions = {}
            self.task_positions = {}
            self.start_location = None
            self.end_location = None
            self.corner_markers = {}
            
            if ids is not None and len(ids) > 0:
                ids = ids.flatten()
                
                # First pass: Extract corner markers for perspective calibration
                for corner, marker_id in zip(corners, ids):
                    center_px = np.mean(corner[0], axis=0)
                    px, py = float(center_px[0]), float(center_px[1])
                    if marker_id in [0, 1, 2, 3]:
                        self.corner_markers[int(marker_id)] = (px, py)
                
                # Try to establish perspective mapping if all 4 corners detected
                if len(self.corner_markers) == 4:
                    pts_src = np.array([
                        self.corner_markers[0], # Top-Left
                        self.corner_markers[1], # Top-Right
                        self.corner_markers[2], # Bottom-Right
                        self.corner_markers[3]  # Bottom-Left
                    ], dtype=np.float32)
                    
                    pts_dst = np.array([
                        [0, 0],
                        [self.workspace_width, 0],
                        [self.workspace_width, self.workspace_height],
                        [0, self.workspace_height]
                    ], dtype=np.float32)
                    
                    self.perspective_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
                else:
                    self.perspective_matrix = None
                    if time.time() - self.last_debug_time > 3.0:
                        print(f"⚠️ [ArUcoDetector] Incomplete workspace boundary! Corners detected: {list(self.corner_markers.keys())}")
                
                # Second pass: Classify and transform marker centers to world coordinates
                for corner, marker_id in zip(corners, ids):
                    center_px = np.mean(corner[0], axis=0)
                    px, py = float(center_px[0]), float(center_px[1])
                    
                    if self.perspective_matrix is not None:
                        pt_pt = np.array([[[px, py]]], dtype=np.float32)
                        pt_trans = cv2.perspectiveTransform(pt_pt, self.perspective_matrix)
                        x_mm, y_mm = float(pt_trans[0][0][0]), float(pt_trans[0][0][1])
                    else:
                        h, w = gray.shape[:2]
                        x_mm = (px / w) * self.workspace_width
                        y_mm = (py / h) * self.workspace_height
                    
                    coord_tuple = (x_mm, y_mm)
                    
                    if marker_id in [100, 101, 102]:
                        self.robot_positions[int(marker_id)] = coord_tuple
                    elif marker_id in [20, 21]:
                        self.task_positions[int(marker_id)] = coord_tuple
                    elif marker_id == 10:
                        self.start_location = coord_tuple
                    elif marker_id == 11:
                        self.end_location = coord_tuple

            if time.time() - self.last_debug_time > 2.0:
                self.last_debug_time = time.time()
                print(f"🔍 [ArUcoDetector] Frame detection status: "
                      f"Robots: {len(self.robot_positions)} (IDs: {list(self.robot_positions.keys())}), "
                      f"Jobs: {len(self.task_positions)} (IDs: {list(self.task_positions.keys())}), "
                      f"START: {'✅' if self.start_location else '❌'}, "
                      f"END: {'✅' if self.end_location else '❌'}")
                
        return self.get_positions()

    def get_positions(self):
        with self.lock:
            return {
                'robots': dict(self.robot_positions),
                'tasks': dict(self.task_positions),
                'start': self.start_location,
                'end': self.end_location,
                'corners': dict(self.corner_markers)
            }
