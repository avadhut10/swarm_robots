import cv2
import numpy as np
import sys

# Load images using raw strings for Windows paths
cube1 = cv2.imread(r'F:\swarm_robots\codes\homography\cube1.png')
cube2 = cv2.imread(r'F:\swarm_robots\codes\homography\cube2.png')

# Safety Check
if cube1 is None or cube2 is None:
    sys.exit("Error: Could not load the images. Check the file paths.")

# Convert images to grayscale
gray1 = cv2.cvtColor(cube1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(cube2, cv2.COLOR_BGR2GRAY)

# --- THE FIX ---
# We lower the fastThreshold (default is 20) to make ORB hyper-sensitive 
# so it can pick up the faint wood grain on your cubes.
orb = cv2.ORB_create(nfeatures=1000, fastThreshold=5, edgeThreshold=5)

# Find keypoints and descriptors
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

if des1 is None or des2 is None:
    sys.exit("Error: Still not enough features found. The images are too smooth.")

# Use BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw the top 20 matches so you can see what it grabbed onto
img_matches = cv2.drawMatches(cube1, kp1, cube2, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

MIN_MATCH_COUNT = 4 

if len(matches) >= MIN_MATCH_COUNT:
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    
    if H is not None:
        height, width, _ = cube2.shape
        cube1_aligned = cv2.warpPerspective(cube1, H, (width, height))

        # Display results
        cv2.imshow('Aligned Image', cube1_aligned)
        cv2.imshow('Matches', img_matches)
        print("Press any key on the image windows to close them...")
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
    else:
        print("Homography could not be computed.")
else:
    print(f"Not enough matches are found - {len(matches)}/{MIN_MATCH_COUNT}")