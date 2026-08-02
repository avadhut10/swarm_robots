import cv2
import numpy as np
import os

def verify_aruco_markers():
    """
    Quick test to verify ArUco markers are visible and detectable
    """
    print("🔍 Verifying ArUco Markers")
    print("="*60)
    print("Place your printed ArUco markers in view of camera")
    print("Press 'q' to quit")
    print("="*60)
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    # ArUco dictionary (matching what you printed)
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    except:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5)
    
    parameters = cv2.aruco.DetectorParameters()
    
    # Your marker IDs
    expected_ids = [0, 1, 2, 3, 10, 11, 20, 21, 100, 101, 102]
    
    print("\n📋 Looking for these markers:")
    print(f"   Boundary:  {[0, 1, 2, 3]}")
    print(f"   Start/Stop: {[10, 11]}")
    print(f"   Jobs:       {[20, 21]}")
    print(f"   Robots:     {[100, 101, 102]}")
    print()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        try:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=parameters
            )
        except:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, aruco_dict
            )
        
        display = frame.copy()
        detected_ids = []
        
        if ids is not None and len(ids) > 0:
            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                detected_ids.append(marker_id)
                
                # Draw marker
                cv2.aruco.drawDetectedMarkers(display, corners, ids)
                
                # Add ID label
                center_x = int(np.mean(corners[i][0][:, 0]))
                center_y = int(np.mean(corners[i][0][:, 1]))
                
                # Color code by type
                if marker_id in [0, 1, 2, 3]:
                    color = (0, 255, 0)  # Green - Boundary
                    label = f"B{marker_id}"
                elif marker_id in [10, 11]:
                    color = (255, 165, 0)  # Orange - Start/Stop
                    label = "Start" if marker_id == 10 else "End"
                elif marker_id in [20, 21]:
                    color = (255, 0, 255)  # Magenta - Jobs
                    label = f"Job{marker_id-19}"
                elif marker_id in [100, 101, 102]:
                    color = (0, 0, 255)  # Red - Robots
                    label = f"R{marker_id-99}"
                else:
                    color = (255, 255, 0)  # Yellow - Unknown
                    label = f"ID{marker_id}"
                
                cv2.putText(display, label, (center_x-20, center_y-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Show detection status
        status_text = f"Found: {len(detected_ids)} markers"
        cv2.putText(display, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show which markers are missing
        missing = [id for id in expected_ids if id not in detected_ids]
        if missing:
            cv2.putText(display, f"Missing: {missing}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.imshow('ArUco Marker Verification', display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n📊 Summary:")
    print(f"   Detected: {len(detected_ids)} markers")
    if len(detected_ids) == len(expected_ids):
        print("   ✅ All markers detected!")
    else:
        print(f"   ⚠️ Missing: {len(expected_ids) - len(detected_ids)} markers")
        print(f"      IDs: {missing}")

if __name__ == "__main__":
    verify_aruco_markers()