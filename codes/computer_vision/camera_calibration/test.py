import cv2
import numpy as np

def test_8x5_chessboard():
    """
    Test detection with your 8x5 chessboard
    """
    print("🎯 Testing 8x5 Chessboard Detection")
    print("="*50)
    print("Your chessboard has 8x5 INTERNAL corners")
    print("This means: 8 squares across, 5 squares down")
    print()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    # YOUR CHESSBOARD SIZE
    chessboard_size = (8, 5)  # 8 internal corners horizontally, 5 vertically
    
    print("📸 Hold your 8x5 chessboard in front of camera")
    print("   Move it around to test detection")
    print("   Press 'q' to quit")
    print()
    
    frame_count = 0
    detected_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect your chessboard
        ret, corners = cv2.findChessboardCorners(
            gray, 
            chessboard_size,  # (8, 5) for your board
            None,
            cv2.CALIB_CB_ADAPTIVE_THRESH + 
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        display = frame.copy()
        
        if ret:
            detected_count += 1
            # Draw detected corners
            cv2.drawChessboardCorners(display, chessboard_size, corners, ret)
            cv2.putText(display, "✅ DETECTED! (8x5)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"Corners found: {len(corners)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display, "❌ NOT DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(display, "Tips:", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(display, "- Ensure full chessboard is visible", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(display, "- Good lighting is essential", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(display, "- Keep chessboard flat", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Show info
        cv2.putText(display, f"Frame: {frame_count}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('8x5 Chessboard Test', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "="*50)
    print(f"📊 Detection Results:")
    print(f"   Frames processed: {frame_count}")
    print(f"   Detected: {detected_count}")
    print(f"   Detection rate: {detected_count/frame_count*100:.1f}%")
    
    if detected_count > 0:
        print("\n✅ Detection works! You can proceed with calibration.")
        print("   Run the calibration script with chessboard_size = (8, 5)")
    else:
        print("\n❌ No detection. Try:")
        print("   1. Adjust lighting (more light, no shadows)")
        print("   2. Move chessboard to center of frame")
        print("   3. Make sure chessboard is not tilted")
        print("   4. Check chessboard is printed correctly")

if __name__ == "__main__":
    test_8x5_chessboard()