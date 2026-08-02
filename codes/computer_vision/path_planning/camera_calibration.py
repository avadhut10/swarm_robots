import cv2
import numpy as np
import time
from datetime import datetime

class CameraCalibrator:
    """
    Camera calibration for Motorola Smart Connect
    Uses 8x5 chessboard (8 internal corners horizontal, 5 vertical)
    """
    
    def __init__(self, camera_id=1):
        """
        Initialize calibrator
        
        Args:
            camera_id: Camera device ID (1 for Motorola Smart Connect)
        """
        self.camera_id = camera_id
        self.cap = None
        self.is_camera_open = False
        
        # YOUR CHESSBOARD PARAMETERS
        self.chessboard_size = (8, 5)  # Internal corners
        self.square_size_mm = 30       # Must match your printed chessboard
        
        # Calibration data
        self.object_points = []
        self.image_points = []
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.calibration_error = 0
        
        # State
        self.captured_frames = 0
        self.required_frames = 25
        self.calibration_complete = False
        self.latest_frame = None
        
        print("\n📐 Camera Calibrator Initialized")
        print("="*60)
        print(f"Chessboard: {self.chessboard_size[0]}x{self.chessboard_size[1]} internal corners")
        print(f"Square size: {self.square_size_mm}mm")
        print(f"Required frames: {self.required_frames}")
        print("="*60)
    
    def open_camera(self):
        """Open Motorola Smart Connect camera"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"❌ Could not open camera {self.camera_id}")
            return False
        
        # Set resolution
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
            print("📷 Camera closed")
    
    def find_chessboard(self, frame):
        """
        Find chessboard corners in frame
        
        Returns:
            (success, corners, frame_with_corners)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            self.chessboard_size,  # (8, 5) for your board
            None,
            cv2.CALIB_CB_ADAPTIVE_THRESH + 
            cv2.CALIB_CB_NORMALIZE_IMAGE + 
            cv2.CALIB_CB_FAST_CHECK
        )
        
        if ret:
            # Refine corners to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw corners
            frame_with_corners = frame.copy()
            cv2.drawChessboardCorners(frame_with_corners, self.chessboard_size, 
                                     corners_refined, ret)
            
            return True, corners_refined, frame_with_corners
        
        return False, None, frame
    
    def capture_frame(self, frame):
        """
        Process and capture a frame for calibration
        
        Returns:
            (success, display_frame)
        """
        success, corners, display_frame = self.find_chessboard(frame)
        
        if success and self.captured_frames < self.required_frames:
            # Create object points for 8x5 chessboard
            objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 
                                  0:self.chessboard_size[1]].T.reshape(-1, 2)
            objp = objp * self.square_size_mm
            
            # Store points
            self.object_points.append(objp)
            self.image_points.append(corners)
            self.captured_frames += 1
            
            # Show capture status
            cv2.putText(display_frame, f"✅ CAPTURED: {self.captured_frames}/{self.required_frames}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Progress bar
            progress = self.captured_frames / self.required_frames
            bar_width = 300
            bar_x = 10
            bar_y = 120
            cv2.rectangle(display_frame, (bar_x, bar_y), 
                         (bar_x + int(bar_width * progress), bar_y + 20), 
                         (0, 255, 0), -1)
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), 
                         (255, 255, 255), 2)
            cv2.putText(display_frame, f"{int(progress*100)}%", 
                       (bar_x + bar_width//2 - 20, bar_y + 16),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return True, display_frame
        
        # Show status when not capturing
        if self.captured_frames < self.required_frames:
            cv2.putText(display_frame, 
                       f"📷 Captured: {self.captured_frames}/{self.required_frames}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if not success:
                cv2.putText(display_frame, "❌ No chessboard detected (8x5)", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_frame, "Hold your 8x5 chessboard flat", (10, 115),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return False, display_frame
    
    def calibrate(self):
        """
        Perform camera calibration using captured frames
        """
        if len(self.object_points) < 10:
            print(f"❌ Not enough calibration data: {len(self.object_points)} frames")
            print("   Need at least 10 frames with detected chessboard")
            return False, None, None, 0
        
        print(f"\n🔧 Calibrating camera with {len(self.object_points)} frames...")
        
        # Get image size from latest frame
        if self.latest_frame is not None:
            h, w = self.latest_frame.shape[:2]
        else:
            h, w = 1080, 1920
        
        # Perform calibration
        ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points,
            self.image_points,
            (w, h),
            None,
            None
        )
        
        if ret:
            # Calculate reprojection error
            total_error = 0
            for i in range(len(self.object_points)):
                img_points2, _ = cv2.projectPoints(
                    self.object_points[i], rvecs[i], tvecs[i], 
                    camera_matrix, distortion_coeffs
                )
                error = cv2.norm(self.image_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
                total_error += error
            
            mean_error = total_error / len(self.object_points)
            
            self.camera_matrix = camera_matrix
            self.distortion_coeffs = distortion_coeffs
            self.calibration_error = mean_error
            self.calibration_complete = True
            
            print(f"\n✅ Calibration successful!")
            print(f"   Reprojection error: {mean_error:.4f} pixels")
            print(f"\n   Camera Matrix:")
            print(camera_matrix)
            print(f"\n   Distortion coefficients: {distortion_coeffs.ravel()}")
            
            # Save calibration
            self.save_calibration()
            
            return True, camera_matrix, distortion_coeffs, mean_error
        
        return False, None, None, 0
    
    def save_calibration(self):
        """Save calibration data to file"""
        if not self.calibration_complete:
            print("❌ No calibration data to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as numpy file
        filename = f'motorola_calibration_{timestamp}.npz'
        np.savez(filename,
                camera_matrix=self.camera_matrix,
                distortion_coeffs=self.distortion_coeffs,
                calibration_error=self.calibration_error,
                chessboard_size=self.chessboard_size,
                square_size_mm=self.square_size_mm,
                frames_used=len(self.object_points))
        
        print(f"\n✅ Calibration saved to: {filename}")
        print(f"   Remember this filename for your detection code!")
        
        # Also save as text for easy viewing
        txt_filename = f'motorola_calibration_{timestamp}.txt'
        with open(txt_filename, 'w') as f:
            f.write("=== CAMERA CALIBRATION DATA ===\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Camera: Motorola Smart Connect (ID: {self.camera_id})\n")
            f.write(f"Chessboard: {self.chessboard_size[0]}x{self.chessboard_size[1]}\n")
            f.write(f"Square size: {self.square_size_mm}mm\n")
            f.write(f"Frames used: {len(self.object_points)}\n")
            f.write(f"Reprojection error: {self.calibration_error:.4f} pixels\n\n")
            f.write("Camera Matrix:\n")
            f.write(str(self.camera_matrix) + "\n\n")
            f.write("Distortion Coefficients:\n")
            f.write(str(self.distortion_coeffs.ravel()) + "\n")
        
        print(f"✅ Text version saved to: {txt_filename}")
    
    def verify_calibration(self):
        """Verify calibration by undistorting a live frame"""
        if not self.calibration_complete:
            print("❌ No calibration to verify")
            return
        
        print("\n📊 Verifying calibration...")
        print("   Hold the chessboard in front of camera")
        print("   Press any key to close verification")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Undistort frame
            h, w = frame.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, 
                self.distortion_coeffs, 
                (w, h), 
                1, 
                (w, h)
            )
            
            undistorted = cv2.undistort(
                frame, 
                self.camera_matrix, 
                self.distortion_coeffs, 
                None, 
                new_camera_matrix
            )
            
            # Crop to valid area
            x, y, w, h = roi
            undistorted = undistorted[y:y+h, x:x+w]
            
            # Resize for display
            display_orig = cv2.resize(frame, (640, 360))
            display_undist = cv2.resize(undistorted, (640, 360))
            
            # Show comparison
            comparison = np.hstack([display_orig, display_undist])
            
            # Add labels
            cv2.putText(comparison, "ORIGINAL", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(comparison, "UNDISTORTED", (650, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(comparison, f"Error: {self.calibration_error:.4f} px", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Calibration Verification', comparison)
            
            if cv2.waitKey(1) != -1:
                break
        
        cv2.destroyWindow('Calibration Verification')
    
    def run_calibration(self):
        """
        Main calibration interface
        """
        if not self.is_camera_open:
            if not self.open_camera():
                return
        
        print("\n" + "="*60)
        print("📱 MOTOROLA SMART CONNECT CALIBRATION")
        print("="*60)
        print(f"CHESSBOARD SIZE: {self.chessboard_size[0]}x{self.chessboard_size[1]}")
        print("="*60)
        print("\nINSTRUCTIONS:")
        print("  1. Hold your 8x5 chessboard in front of camera")
        print("  2. Move it to different positions and angles:")
        print("     - Center, corners, tilted")
        print("     - Close (30cm), medium (50cm), far (100cm)")
        print("     - Rotate horizontally and vertically")
        print("  3. Press SPACE to capture current view (when detected)")
        print("  4. Press 'c' to calculate calibration")
        print("  5. Press 'v' to verify calibration")
        print("  6. Press 'q' to quit")
        print("="*60)
        print(f"\nCaptured: {self.captured_frames}/{self.required_frames}")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            self.latest_frame = frame.copy()
            
            # Process frame
            success, display_frame = self.capture_frame(frame)
            
            # Add overlays
            h, w = display_frame.shape[:2]
            
            # Camera info
            cv2.putText(display_frame, "📱 Motorola Smart Connect", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Chessboard: {self.chessboard_size[0]}x{self.chessboard_size[1]}", 
                       (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Calibration status
            if self.calibration_complete:
                cv2.putText(display_frame, "✅ CALIBRATED", (w - 200, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Error: {self.calibration_error:.3f}px", (w - 200, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(display_frame, "⏳ NOT CALIBRATED", (w - 220, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show display
            cv2.imshow('Motorola Camera Calibration', display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == 32:  # SPACE
                if success:
                    print(f"✅ Captured frame {self.captured_frames}/{self.required_frames}")
                    if self.captured_frames >= self.required_frames:
                        print("🎯 All frames captured! Press 'c' to calibrate")
                else:
                    print("❌ No chessboard detected - try adjusting position")
            
            elif key == ord('c'):
                if self.captured_frames >= 10:
                    success, _, _, error = self.calibrate()
                    if success:
                        print("✅ Calibration complete!")
                    else:
                        print("❌ Calibration failed - try capturing more frames")
                else:
                    print(f"❌ Need more frames: {self.captured_frames}/10 minimum")
            
            elif key == ord('v'):
                if self.calibration_complete:
                    self.verify_calibration()
                else:
                    print("❌ Calibrate first (press 'c')")
            
            elif key == ord('r'):
                self.object_points = []
                self.image_points = []
                self.captured_frames = 0
                self.calibration_complete = False
                print("🔄 Reset calibration data")
        
        # Cleanup
        self.close_camera()
        cv2.destroyAllWindows()
        
        print("\n📐 Calibration complete!")
        if self.calibration_complete:
            print(f"   Calibration error: {self.calibration_error:.4f} pixels")
            if self.calibration_error < 0.5:
                print("   ✅ EXCELLENT calibration!")
            elif self.calibration_error < 1.0:
                print("   ✅ Good calibration")
            else:
                print("   ⚠️ Consider re-calibrating for better accuracy")
        else:
            print("   No calibration completed")


# Main execution
if __name__ == "__main__":
    print("🚀 MOTOROLA SMART CONNECT CALIBRATION")
    print("="*60)
    
    # Create calibrator with your camera ID
    calibrator = CameraCalibrator(camera_id=1)
    
    # Run calibration
    calibrator.run_calibration()