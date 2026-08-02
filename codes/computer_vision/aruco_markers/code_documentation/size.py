# Save code to visual_check.py and run to generate visual overlay map
import fitz # PyMuPDF
import matplotlib.pyplot as plt
import matplotlib.patches as patches

doc = fitz.open("aruco_markers_ALL_ON_ONE_SHEET.pdf")
fig, axs = plt.subplots(2, 2, figsize=(10, 14))
axs = axs.flatten()

for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=100)
    width, height = page.rect.width, page.rect.height # Points (1 pt = 1/72 inch)
    
    # Render page background
    img_data = pix.tobytes("png")
    axs[i].imshow(plt.imread(fitz.io.BytesIO(img_data)))
    
    # Map vector paths to verify exact geometric boxes
    paths = page.get_drawings()
    for path in paths:
        if path["type"] == "s" or path["type"] == "p": # Rectangles/Polygons
            rect = path["rect"]
            # Convert points to mm for real physical verification
            w_mm = (rect.width / 72) * 25.4
            h_mm = (rect.height / 72) * 25.4
            
            if 30 <= w_mm <= 90: # Filter for marker dimensions
                # Draw visual verification box over image
                rect_patch = patches.Rectangle(
                    (rect.x0 * (pix.width/width), rect.y0 * (pix.height/height)),
                    rect.width * (pix.width/width), rect.height * (pix.height/height),
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                axs[i].add_patch(rect_patch)
                axs[i].text(rect.x0 * (pix.width/width), (rect.y0 - 10) * (pix.height/height), 
                            f"{w_mm:.1f}mm x {h_mm:.1f}mm", color='red', fontsize=9, weight='bold')

    axs[i].set_title(f"Page {i+1} Geometric Verification")
    axs[i].axis('off')

plt.tight_layout()
plt.savefig("marker_visual_verification.png", dpi=150)
plt.show()