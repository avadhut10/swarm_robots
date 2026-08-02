import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
import os
import sys

class ArUcoGenerator:
    def __init__(self):
        # Check OpenCV version
        self.opencv_version = cv2.__version__
        print(f"OpenCV Version: {self.opencv_version}")
        
        # Use DICT_5X5_1000 for more IDs
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
            print("Using DICT_5X5_1000 (supports up to 1023 IDs)")
        except:
            try:
                self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5)
                print("Using DICT_5X5")
            except:
                self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5)
                print("Using legacy DICT_5X5")
        
        # Marker sizes in mm
        self.marker_size_mm = {
            'boundary': 80,
            'start_stop': 60,
            'robot': 40,
            'job': 50
        }
        
        # ID assignments
        self.id_groups = {
            'boundary': [0, 1, 2, 3],
            'start_stop': [10, 11],
            'job': [20, 21],
            'robot': [100, 101, 102]
        }
        
        # DPI for printing (300 DPI gives good quality)
        self.dpi = 300
        # Convert mm to pixels at 300 DPI (1 inch = 25.4 mm)
        self.pixels_per_mm = self.dpi / 25.4  # ~11.81 pixels per mm at 300 DPI

    def generate_aruco_marker(self, marker_id, size_mm, add_border=True, border_mm=5):
        """Generate a single ArUco marker with exact physical size"""
        # Calculate pixel size at 300 DPI
        size_pixels = int(size_mm * self.pixels_per_mm)
        
        # Generate the marker
        try:
            marker_img = cv2.aruco.generateImageMarker(self.aruco_dict, marker_id, size_pixels)
        except:
            marker_img = cv2.aruco.drawMarker(self.aruco_dict, marker_id, size_pixels)
        
        if add_border:
            # Add white border
            border_pixels = int(border_mm * self.pixels_per_mm)
            final_img = np.ones((size_pixels + 2*border_pixels, 
                               size_pixels + 2*border_pixels), dtype=np.uint8) * 255
            final_img[border_pixels:border_pixels+size_pixels, 
                     border_pixels:border_pixels+size_pixels] = marker_img
        else:
            final_img = marker_img
        
        # Convert to RGB for text
        final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(final_img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # Add ID text and size information
        try:
            font = ImageFont.truetype("arial.ttf", int(size_mm * 0.4))
        except:
            font = ImageFont.load_default()
        
        # Add text below the marker
        text = f"ID: {marker_id}  |  {size_mm}mm"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        
        # Create image with space for text
        text_height = int(15 * self.pixels_per_mm)
        new_height = final_img.shape[0] + text_height
        new_img = np.ones((new_height, final_img.shape[1], 3), dtype=np.uint8) * 255
        new_img[:final_img.shape[0], :, :] = final_img_rgb
        
        # Add text
        pil_new_img = Image.fromarray(new_img)
        draw_new = ImageDraw.Draw(pil_new_img)
        
        text_x = (new_img.shape[1] - text_width) // 2
        text_y = final_img.shape[0] + int(5 * self.pixels_per_mm)
        draw_new.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
        
        # Add a scale bar (1cm ruler) at the bottom
        scale_bar_length_mm = 10  # 1cm
        scale_bar_pixels = int(scale_bar_length_mm * self.pixels_per_mm)
        bar_y = new_height - int(8 * self.pixels_per_mm)
        bar_x_start = (new_img.shape[1] - scale_bar_pixels) // 2
        bar_x_end = bar_x_start + scale_bar_pixels
        
        # Draw scale bar
        draw_new.rectangle([bar_x_start, bar_y, bar_x_end, bar_y + int(2 * self.pixels_per_mm)], 
                          fill=(0, 0, 0))
        draw_new.text((bar_x_start, bar_y - int(10 * self.pixels_per_mm)), 
                     "10mm", fill=(0, 0, 0), font=font)
        
        return np.array(pil_new_img)

    def generate_all_markers(self):
        """Generate all markers with exact physical sizes"""
        print("\n" + "="*60)
        print("Generating ArUco markers with EXACT physical sizes...")
        print(f"Resolution: {self.dpi} DPI ({self.pixels_per_mm:.2f} pixels/mm)")
        print("="*60)
        
        # Create directories
        if not os.path.exists('aruco_markers'):
            os.makedirs('aruco_markers')
        
        all_markers = []
        
        # Generate boundary markers
        for marker_id in self.id_groups['boundary']:
            size_mm = self.marker_size_mm['boundary']
            marker_img = self.generate_aruco_marker(marker_id, size_mm)
            filename = f"aruco_markers/boundary_ID{marker_id}_{size_mm}mm.png"
            cv2.imwrite(filename, cv2.cvtColor(marker_img, cv2.COLOR_RGB2BGR), 
                       [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"✓ Boundary ID {marker_id}: {size_mm}mm -> {size_mm*self.pixels_per_mm:.0f}x{size_mm*self.pixels_per_mm:.0f}px")
            all_markers.append((marker_id, size_mm, 'boundary', filename))
        
        # Generate start/stop markers
        for marker_id in self.id_groups['start_stop']:
            size_mm = self.marker_size_mm['start_stop']
            marker_img = self.generate_aruco_marker(marker_id, size_mm)
            filename = f"aruco_markers/startstop_ID{marker_id}_{size_mm}mm.png"
            cv2.imwrite(filename, cv2.cvtColor(marker_img, cv2.COLOR_RGB2BGR),
                       [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"✓ Start/Stop ID {marker_id}: {size_mm}mm -> {size_mm*self.pixels_per_mm:.0f}x{size_mm*self.pixels_per_mm:.0f}px")
            all_markers.append((marker_id, size_mm, 'start_stop', filename))
        
        # Generate job markers
        for marker_id in self.id_groups['job']:
            size_mm = self.marker_size_mm['job']
            marker_img = self.generate_aruco_marker(marker_id, size_mm)
            filename = f"aruco_markers/job_ID{marker_id}_{size_mm}mm.png"
            cv2.imwrite(filename, cv2.cvtColor(marker_img, cv2.COLOR_RGB2BGR),
                       [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"✓ Job ID {marker_id}: {size_mm}mm -> {size_mm*self.pixels_per_mm:.0f}x{size_mm*self.pixels_per_mm:.0f}px")
            all_markers.append((marker_id, size_mm, 'job', filename))
        
        # Generate robot markers
        for marker_id in self.id_groups['robot']:
            size_mm = self.marker_size_mm['robot']
            marker_img = self.generate_aruco_marker(marker_id, size_mm)
            filename = f"aruco_markers/robot_ID{marker_id}_{size_mm}mm.png"
            cv2.imwrite(filename, cv2.cvtColor(marker_img, cv2.COLOR_RGB2BGR),
                       [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"✓ Robot ID {marker_id}: {size_mm}mm -> {size_mm*self.pixels_per_mm:.0f}x{size_mm*self.pixels_per_mm:.0f}px")
            all_markers.append((marker_id, size_mm, 'robot', filename))
        
        print("="*60)
        print(f"✅ Total markers generated: {len(all_markers)}")
        
        # Create measurement guide
        self.create_measurement_guide()
        
        # Create printable HTML with all markers
        self.create_printable_html()
        
        # Create a simple printable sheet with all markers
        self.create_printable_sheet(all_markers)
        
        return all_markers

    def create_printable_sheet(self, all_markers):
        """Create a single printable sheet with all markers arranged"""
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm
        from reportlab.lib.utils import ImageReader
        
        print("\n📄 Creating printable sheet with all markers...")
        
        # Create PDF
        c = canvas.Canvas("aruco_markers_ALL_ON_ONE_SHEET.pdf", pagesize=A4)
        page_width, page_height = A4
        
        # Layout: 4 columns, markers arranged by type
        margin = 10 * mm
        available_width = page_width - 2 * margin
        available_height = page_height - 2 * margin
        
        # Define marker positions (organized layout)
        # Row 1: Boundary markers (4 markers)
        # Row 2: Start/Stop (2 markers) + Job (2 markers)
        # Row 3: Robot markers (3 markers)
        
        marker_size = 45 * mm  # Size for each marker on the sheet
        
        # Group markers by type
        groups = {
            'boundary': [(mid, size, path) for mid, size, group, path in all_markers if group == 'boundary'],
            'start_stop': [(mid, size, path) for mid, size, group, path in all_markers if group == 'start_stop'],
            'job': [(mid, size, path) for mid, size, group, path in all_markers if group == 'job'],
            'robot': [(mid, size, path) for mid, size, group, path in all_markers if group == 'robot']
        }
        
        # Sort each group by ID
        for group in groups:
            groups[group].sort(key=lambda x: x[0])
        
        y_position = page_height - margin
        
        # Draw each group
        for group_name, markers in groups.items():
            if not markers:
                continue
            
            # Group header
            c.setFont("Helvetica-Bold", 12)
            header = f"{group_name.upper()} MARKERS"
            c.drawString(margin, y_position, header)
            y_position -= 5 * mm
            
            # Calculate layout for this group
            num_markers = len(markers)
            if num_markers <= 4:
                cols = num_markers
            else:
                cols = 4
            
            spacing = (available_width - cols * marker_size) / (cols + 1)
            
            # Place markers in row
            for idx, (marker_id, size_mm, img_path) in enumerate(markers):
                col = idx % cols
                x = margin + spacing + col * (marker_size + spacing)
                
                # Draw marker
                img = ImageReader(img_path)
                c.drawImage(img, x, y_position - marker_size, 
                           width=marker_size, height=marker_size)
                
                # Add label below
                c.setFont("Helvetica", 8)
                label = f"ID:{marker_id} ({size_mm}mm)"
                c.drawString(x, y_position - marker_size - 4 * mm, label)
            
            y_position -= (marker_size + 10 * mm)
            
            # Check if we need new page
            if y_position < margin:
                c.showPage()
                y_position = page_height - margin
        
        c.save()
        print("✅ Created: aruco_markers_ALL_ON_ONE_SHEET.pdf")
        print("   All markers on a single sheet for easy printing")

    def create_printable_html(self):
        """Create an HTML file for easy printing of all markers"""
        html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ArUco Markers for Printing</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; }
        .header { 
            background: #2c3e50; 
            color: white; 
            padding: 20px; 
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .warning { 
            background: #fff3cd; 
            border: 1px solid #ffc107; 
            padding: 15px; 
            margin: 20px 0;
            border-radius: 5px;
        }
        .warning strong { color: #856404; }
        .marker-group { 
            border: 2px solid #ddd; 
            padding: 20px; 
            margin: 20px 0;
            border-radius: 10px;
            background: white;
            page-break-after: always;
        }
        .marker-group h2 { 
            background: #ecf0f1; 
            padding: 10px; 
            margin: -20px -20px 20px -20px;
            border-radius: 8px 8px 0 0;
        }
        .marker-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 30px;
            justify-items: center;
        }
        .marker-item {
            text-align: center;
            padding: 10px;
        }
        .marker-item img {
            max-width: 150px;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .marker-item .label {
            margin-top: 10px;
            font-weight: bold;
            font-size: 14px;
        }
        .marker-item .size {
            color: #666;
            font-size: 12px;
        }
        .button {
            background: #3498db;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .button:hover { background: #2980b9; }
        @media print {
            .no-print { display: none; }
            .marker-group { page-break-after: always; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ArUco Markers for Swarm Robot System</h1>
            <p>Dictionary: DICT_5X5_1000 | 300 DPI | Supports IDs 0-1023</p>
        </div>
        
        <div class="warning no-print">
            <strong>IMPORTANT:</strong> 
            Print at 100% scale (no scaling/fit-to-page). 
            Each marker has a 10mm scale bar for verification.
            <br><br>
            <button class="button" onclick="window.print()">🖨 Print This Page</button>
        </div>
"""
        
        # Add marker groups
        groups = [
            ('boundary', 'BOUNDARY MARKERS - 80mm', [0, 1, 2, 3]),
            ('startstop', 'START/STOP MARKERS - 60mm', [10, 11]),
            ('job', 'JOB LOCATIONS - 50mm', [20, 21]),
            ('robot', 'ROBOT MARKERS - 40mm', [100, 101, 102])
        ]
        
        for group_name, title, ids in groups:
            html_content += f"""
        <div class="marker-group">
            <h2>{title}</h2>
            <div class="marker-grid">
"""
            for marker_id in ids:
                # Find the correct size for this ID
                if marker_id in [0, 1, 2, 3]:
                    size = 80
                elif marker_id in [10, 11]:
                    size = 60
                elif marker_id in [20, 21]:
                    size = 50
                else:
                    size = 40
                
                html_content += f"""
                <div class="marker-item">
                    <img src="aruco_markers/{group_name}_ID{marker_id}_{size}mm.png" alt="Marker ID {marker_id}">
                    <div class="label">ID: {marker_id}</div>
                    <div class="size">{size}mm x {size}mm</div>
                </div>
"""
            
            html_content += """
            </div>
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        # Write with utf-8 encoding
        with open("aruco_markers_printable.html", "w", encoding='utf-8') as f:
            f.write(html_content)
        
        print("\n✅ Printable HTML page saved to: aruco_markers_printable.html")
        print("   Open in browser and click Print or press Ctrl+P")
        print("   IMPORTANT: Set print scale to 100%!")

    def create_measurement_guide(self):
        """Create a detailed measurement guide"""
        guide = """
        ================================================================
        ARUCO MARKER MEASUREMENT GUIDE
        ================================================================
        
        GENERATION SETTINGS:
        -------------------
        Resolution: 300 DPI
        Pixel density: 11.81 pixels/mm
        Dictionary: DICT_5X5_1000 (IDs 0-1023)
        
        MARKER SIZES AND PIXEL DIMENSIONS:
        ----------------------------------
        Boundary Markers (IDs 0-3):  80mm x 80mm  -> 945 x 945 pixels
        Start/Stop (IDs 10-11):      60mm x 60mm  -> 709 x 709 pixels
        Job Locations (IDs 20-21):   50mm x 50mm  -> 591 x 591 pixels
        Robot Markers (IDs 100-102): 40mm x 40mm  -> 472 x 472 pixels
        
        HOW TO MEASURE CORRECTLY:
        -------------------------
        1. OPEN 'aruco_markers_ALL_ON_ONE_SHEET.pdf' or 'aruco_markers_printable.html'
        2. Set printer to 100% scale (NO scaling/fit-to-page)
        3. Print on A4 paper
        4. After printing, measure with a ruler:
           - The 80mm marker should be exactly 80mm x 80mm
           - The 60mm marker should be exactly 60mm x 60mm
           - The 50mm marker should be exactly 50mm x 50mm
           - The 40mm marker should be exactly 40mm x 40mm
        5. Use the 10mm scale bar on each marker to verify
        
        IF SIZES ARE WRONG:
        -------------------
        If the printed size is not exact:
        1. Check printer settings - ensure 100% scale
        2. Disable "Fit to Page" or "Shrink to Fit"
        3. Check "Actual Size" option in print dialog
        4. If still wrong, measure the actual size and use THAT value in your code
        
        FOR DISTANCE CALCULATION:
        -------------------------
        Use these values in your detection code:
        
        MARKER_SIZES_MM = {
            0: 80,  1: 80,  2: 80,  3: 80,      # Boundary
            10: 60, 11: 60,                     # Start/Stop
            20: 50, 21: 50,                     # Job
            100: 40, 101: 40, 102: 40           # Robots
        }
        
        IMPORTANT: If your printer scales differently, measure each marker
        and update these values with your actual measured sizes!
        
        ================================================================
        """
        
        with open("aruco_measurement_guide.txt", "w", encoding='utf-8') as f:
            f.write(guide)
        
        print("\n✅ Measurement guide saved to: aruco_measurement_guide.txt")

    def visualize_layout(self):
        """Create a visual layout plan"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        workspace_width = 2000
        workspace_height = 1500
        
        # Draw workspace
        ax.add_patch(Rectangle((0, 0), workspace_width, workspace_height, 
                              fill=False, edgecolor='black', linewidth=2))
        
        # Marker positions
        markers = {
            'ID0 (80mm)': (50, workspace_height-50),
            'ID1 (80mm)': (workspace_width-50, workspace_height-50),
            'ID2 (80mm)': (50, 50),
            'ID3 (80mm)': (workspace_width-50, 50),
            'ID10 (Start)': (workspace_width/2 - 100, workspace_height-200),
            'ID11 (End)': (workspace_width/2 + 100, 200),
            'ID20 (Job1)': (workspace_width/2 - 150, workspace_height/2 + 100),
            'ID21 (Job2)': (workspace_width/2 + 150, workspace_height/2 - 100),
            'ID100 (Robot1)': (300, workspace_height/2),
            'ID101 (Robot2)': (workspace_width/2, workspace_height/2),
            'ID102 (Robot3)': (workspace_width-300, workspace_height/2),
        }
        
        for label, (x, y) in markers.items():
            ax.scatter(x, y, s=200, color='red', marker='s')
            ax.annotate(label, (x, y), xytext=(10, 10), 
                       textcoords='offset points', fontsize=8)
        
        ax.set_xlim(-100, workspace_width+100)
        ax.set_ylim(-100, workspace_height+100)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('ArUco Marker Layout Plan')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig('aruco_layout_plan.png', dpi=300, bbox_inches='tight')
        print("✅ Layout plan saved as: aruco_layout_plan.png")
        plt.show()

def generate_chessboard(output_file="chessboard_calibration.pdf", squares_x=9, squares_y=6, square_size_mm=25):
    """Generate chessboard for camera calibration"""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    
    print(f"\n📐 Generating chessboard for camera calibration...")
    
    # Create PDF
    c = canvas.Canvas(output_file, pagesize=A4)
    page_width, page_height = A4
    
    # Calculate board size
    board_width = squares_x * square_size_mm * mm
    board_height = squares_y * square_size_mm * mm
    
    # Center the board on page
    x_offset = (page_width - board_width) / 2
    y_offset = (page_height - board_height) / 2
    
    # Draw chessboard
    for i in range(squares_y):
        for j in range(squares_x):
            x = x_offset + j * square_size_mm * mm
            y = y_offset + i * square_size_mm * mm
            
            # Set color
            if (i + j) % 2 == 0:
                c.setFillColor(colors.black)
            else:
                c.setFillColor(colors.white)
            
            # Draw rectangle
            c.rect(x, y, square_size_mm * mm, square_size_mm * mm, 
                   fill=True, stroke=True)
            c.setStrokeColor(colors.black)
            c.setLineWidth(0.5)
    
    # Add info text
    c.setFont("Helvetica", 10)
    c.drawString(x_offset, y_offset - 20 * mm, 
                f"Chessboard: {squares_x}x{squares_y} squares, {square_size_mm}mm each")
    c.drawString(x_offset, y_offset - 25 * mm, 
                "Use this for camera calibration")
    
    c.save()
    print(f"✅ Chessboard generated: {output_file}")
    print(f"   Size: {squares_x}x{squares_y} squares, each {square_size_mm}mm")
    print(f"   Board dimensions: {board_width/mm:.0f}mm x {board_height/mm:.0f}mm")

# Main execution
if __name__ == "__main__":
    print("🚀 Generating ArUco markers with EXACT physical sizes...")
    print("="*50)
    
    # Generate chessboard
    generate_chessboard("chessboard_calibration.pdf", squares_x=9, squares_y=6, square_size_mm=25)
    
    # Create ArUco generator
    print("\n🎯 Generating ArUco markers...")
    generator = ArUcoGenerator()
    
    # Generate all markers with exact sizes
    markers = generator.generate_all_markers()
    
    # Create layout visualization
    print("\n📋 Creating workspace layout plan...")
    generator.visualize_layout()
    
    print("\n" + "="*50)
    print("✅ GENERATION COMPLETE!")
    print("="*50)
    print("\n📁 FILES CREATED:")
    print("   📄 chessboard_calibration.pdf - Print for camera calibration")
    print("   📄 aruco_markers_ALL_ON_ONE_SHEET.pdf - All markers on one sheet")
    print("   📄 aruco_markers_printable.html - Open in browser and print")
    print("   📄 aruco_measurement_guide.txt - Detailed measurement info")
    print("   📄 aruco_layout_plan.png - Visual layout guide")
    print("   📁 aruco_markers/ - Individual PNG files (300 DPI)")
    
    print("\n📝 CRITICAL PRINTING INSTRUCTIONS:")
    print("   1. OPTION A: Open 'aruco_markers_ALL_ON_ONE_SHEET.pdf'")
    print("   2. OPTION B: Open 'aruco_markers_printable.html' in browser")
    print("   3. In print dialog, set Scale to 100%")
    print("   4. DISABLE 'Fit to Page' or 'Shrink to Fit'")
    print("   5. Select 'Actual Size' if available")
    print("   6. Print on A4 paper")
    print("   7. MEASURE with a ruler - markers should be exact sizes")
    print("   8. If sizes are off, measure actual and update your code!")
    
    print("\n💡 Each marker has a 10mm scale bar for verification!")
    print("="*50)