import numpy as np
import heapq
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

class InteractiveGridPathfinder:
    def __init__(self, rows: int = 10, cols: int = 10):
        """Initialize interactive grid for pathfinding with custom size"""
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)
        self.obstacles = set()
        self.point_a = None
        self.point_b = None
        self.path = []
        
        # Setup the figure and axes
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.2)
        
        # Current mode for clicking
        self.current_mode = 'Point A'  # Can be 'Point A', 'Point B', 'Obstacle', 'Remove'
        
        self.setup_ui()
        self.draw_grid()
        self.connect_events()
        
    def setup_ui(self):
        """Setup interactive UI buttons"""
        # Button axes positions [left, bottom, width, height]
        ax_point_a = plt.axes([0.1, 0.05, 0.15, 0.05])
        ax_point_b = plt.axes([0.27, 0.05, 0.15, 0.05])
        ax_obstacle = plt.axes([0.44, 0.05, 0.15, 0.05])
        ax_remove = plt.axes([0.61, 0.05, 0.15, 0.05])
        ax_clear = plt.axes([0.78, 0.05, 0.1, 0.05])
        
        # Create buttons
        self.btn_a = Button(ax_point_a, 'Set Point A', color='lightgreen')
        self.btn_b = Button(ax_point_b, 'Set Point B', color='lightcoral')
        self.btn_obstacle = Button(ax_obstacle, 'Add Obstacle', color='gray')
        self.btn_remove = Button(ax_remove, 'Remove', color='lightyellow')
        self.btn_clear = Button(ax_clear, 'Clear All', color='lightblue')
        
        # Button callbacks
        self.btn_a.on_clicked(lambda x: self.set_mode('Point A'))
        self.btn_b.on_clicked(lambda x: self.set_mode('Point B'))
        self.btn_obstacle.on_clicked(lambda x: self.set_mode('Obstacle'))
        self.btn_remove.on_clicked(lambda x: self.set_mode('Remove'))
        self.btn_clear.on_clicked(lambda x: self.clear_all())
        
        # Status text
        self.status_text = self.fig.text(0.5, 0.01, f"Grid: {self.rows}x{self.cols} | Mode: {self.current_mode} - Click on grid", 
                                          ha='center', fontsize=12, fontweight='bold')
        
    def set_mode(self, mode):
        """Change the current editing mode"""
        self.current_mode = mode
        self.status_text.set_text(f"Grid: {self.rows}x{self.cols} | Mode: {mode} - Click on grid to place")
        self.fig.canvas.draw_idle()
        print(f"\n🖱️  Mode switched to: {mode}")
        print("   Click on the grid to place your selection")
        
    def connect_events(self):
        """Connect mouse click events"""
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def on_click(self, event):
        """Handle mouse clicks on the grid"""
        if event.inaxes != self.ax:
            return
        
        # Get clicked cell coordinates
        col = int(event.xdata)
        row = (self.rows - 1) - int(event.ydata)  # Convert from plot coordinates to grid coordinates
        
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return
        
        print(f"\n📍 Clicked cell: ({row}, {col})")
        
        if self.current_mode == 'Point A':
            self.point_a = (row, col)
            print(f"✅ Point A set to: {self.point_a}")
            
        elif self.current_mode == 'Point B':
            self.point_b = (row, col)
            print(f"✅ Point B set to: {self.point_b}")
            
        elif self.current_mode == 'Obstacle':
            if (row, col) != self.point_a and (row, col) != self.point_b:
                self.obstacles.add((row, col))
                self.grid[row, col] = 1
                print(f"🚧 Obstacle added at: ({row}, {col})")
            else:
                print(f"⚠️  Cannot place obstacle on Point A or Point B!")
                
        elif self.current_mode == 'Remove':
            if (row, col) in self.obstacles:
                self.obstacles.remove((row, col))
                self.grid[row, col] = 0
                print(f"🗑️  Obstacle removed from: ({row}, {col})")
            elif (row, col) == self.point_a:
                self.point_a = None
                print(f"🗑️  Point A removed")
            elif (row, col) == self.point_b:
                self.point_b = None
                print(f"🗑️  Point B removed")
            else:
                print(f"ℹ️  Nothing to remove at ({row}, {col})")
        
        # Auto-calculate path if both points are set
        if self.point_a and self.point_b:
            self.find_path()
        else:
            self.path = []
            
        self.draw_grid()
        
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid adjacent cells"""
        neighbors = []
        row, col = position
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                if (new_row, new_col) not in self.obstacles:
                    neighbors.append((new_row, new_col))
                    
        return neighbors
    
    def find_path(self):
        """A* algorithm to find shortest path"""
        if not self.point_a or not self.point_b:
            return
        
        start = self.point_a
        end = self.point_b
        
        # Check if start or end is obstacle
        if start in self.obstacles or end in self.obstacles:
            print("❌ Error: Point A or Point B is on an obstacle!")
            self.path = []
            return
        
        open_set = [(0, start)]
        heapq.heapify(open_set)
        
        came_from = {}
        g_score = {start: 0}
        open_set_hash = {start}
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                self.path = path
                self.print_path_stats()
                return
            
            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, end)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f, neighbor))
                        open_set_hash.add(neighbor)
        
        print("❌ No path found! Obstacles might be blocking the way.")
        self.path = []
    
    def print_path_stats(self):
        """Print path statistics"""
        if self.path:
            print("\n" + "="*50)
            print("✅ SHORTEST PATH FOUND!")
            print("="*50)
            print(f"📍 Point A: {self.point_a}")
            print(f"📍 Point B: {self.point_b}")
            print(f"📏 Path Length: {len(self.path)-1} steps")
            print(f"🗺️  Path: {self.path}")
            
            # Print directions
            print("\n🧭 Directions:")
            for i in range(len(self.path) - 1):
                dr = self.path[i+1][0] - self.path[i][0]
                dc = self.path[i+1][1] - self.path[i][1]
                
                if dr == 1: direction = "⬇️  DOWN"
                elif dr == -1: direction = "⬆️  UP"
                elif dc == 1: direction = "➡️  RIGHT"
                elif dc == -1: direction = "⬅️  LEFT"
                
                print(f"  Step {i+1}: {self.path[i]} → {self.path[i+1]} ({direction})")
            print("="*50 + "\n")
    
    def draw_grid(self):
        """Draw the grid with current state"""
        self.ax.clear()
        
        # Calculate font size based on grid size
        if self.rows <= 10:
            font_size = 7
        elif self.rows <= 20:
            font_size = 6
        elif self.rows <= 30:
            font_size = 5
        else:
            font_size = 4
        
        # Draw grid cells
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) in self.obstacles:
                    color = 'black'
                    alpha = 0.7
                    text_color = 'white'
                else:
                    color = 'white'
                    alpha = 0.3
                    text_color = 'black'
                    
                rect = Rectangle((j, self.rows - i - 1), 1, 1, 
                               facecolor=color, edgecolor='gray', 
                               alpha=alpha, linewidth=0.5 if self.rows > 20 else 1)
                self.ax.add_patch(rect)
                
                # Only show coordinates for smaller grids to avoid clutter
                if self.rows <= 15 and self.cols <= 15:
                    self.ax.text(j + 0.5, self.rows - i - 0.5, f'({i},{j})', 
                               ha='center', va='center', fontsize=font_size, 
                               color=text_color, alpha=0.6)
        
        # Draw path
        if self.path:
            path_visual = [(p[1] + 0.5, self.rows - p[0] - 0.5) for p in self.path]
            xs, ys = zip(*path_visual)
            marker_size = max(3, 10 - self.rows * 0.3)
            self.ax.plot(xs, ys, 'b-', linewidth=max(1, 3 - self.rows * 0.1), 
                        label='Shortest Path', zorder=4, alpha=0.8)
            self.ax.plot(xs, ys, 'bo', markersize=marker_size, zorder=4)
        
        # Draw Point A
        if self.point_a:
            marker_size = max(8, 20 - self.rows * 0.5)
            self.ax.plot(self.point_a[1] + 0.5, self.rows - self.point_a[0] - 0.5, 
                       'go', markersize=marker_size, markeredgecolor='darkgreen', 
                       markeredgewidth=2, label='Point A (Start)', zorder=5)
            if self.rows <= 30:
                self.ax.text(self.point_a[1] + 0.5, self.rows - self.point_a[0] - 0.3, 
                           'A', ha='center', va='bottom', fontweight='bold', 
                           fontsize=min(14, 20 - self.rows * 0.3), color='darkgreen')
        
        # Draw Point B
        if self.point_b:
            marker_size = max(8, 20 - self.rows * 0.5)
            self.ax.plot(self.point_b[1] + 0.5, self.rows - self.point_b[0] - 0.5, 
                       'ro', markersize=marker_size, markeredgecolor='darkred', 
                       markeredgewidth=2, label='Point B (End)', zorder=5)
            if self.rows <= 30:
                self.ax.text(self.point_b[1] + 0.5, self.rows - self.point_b[0] - 0.3, 
                           'B', ha='center', va='bottom', fontweight='bold', 
                           fontsize=min(14, 20 - self.rows * 0.3), color='darkred')
        
        # Grid settings
        self.ax.set_xlim(0, self.cols)
        self.ax.set_ylim(0, self.rows)
        self.ax.set_xticks(range(self.cols))
        self.ax.set_yticks(range(self.rows))
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        
        title = f'{self.rows}x{self.cols} Grid - Click to Set Points & Obstacles'
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Show obstacle count and path info
        info_text = f'Obstacles: {len(self.obstacles)}'
        if self.path:
            info_text += f' | Path Length: {len(self.path)-1} steps'
        elif self.point_a and self.point_b:
            info_text += ' | ⚠️  No valid path!'
        self.ax.text(0.5, -0.1, info_text, transform=self.ax.transAxes, 
                    ha='center', fontsize=10, style='italic')
        
        self.fig.canvas.draw_idle()
    
    def clear_all(self):
        """Clear all points, obstacles, and path"""
        self.point_a = None
        self.point_b = None
        self.obstacles.clear()
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.path = []
        print("\n🗑️  Cleared all points and obstacles!")
        self.draw_grid()
    
    def show(self):
        """Display the interactive grid"""
        print("\n" + "="*60)
        print(f"🖱️  INTERACTIVE {self.rows}x{self.cols} GRID PATHFINDER")
        print("="*60)
        print("📋 Instructions:")
        print("   1. Click 'Set Point A' button, then click grid for start point")
        print("   2. Click 'Set Point B' button, then click grid for end point")
        print("   3. Click 'Add Obstacle' button, then click grid to add obstacles")
        print("   4. Click 'Remove' to delete points or obstacles")
        print("   5. Path updates automatically when A and B are set!")
        print("   6. Use 'Clear All' to reset everything")
        print("="*60)
        print("\n🎯 Ready! Start clicking on the grid...\n")
        
        plt.show()


def get_grid_size():
    """Ask user for grid dimensions"""
    print("\n" + "="*60)
    print("🎯 INTERACTIVE GRID PATHFINDER SETUP")
    print("="*60)
    print("\n📐 Enter grid dimensions (recommended: 5-30)")
    print("   Note: Larger grids may be slower to render\n")
    
    while True:
        try:
            rows = int(input("Enter number of rows (e.g., 10): "))
            if rows < 2:
                print("❌ Grid must have at least 2 rows. Try again.\n")
                continue
            if rows > 50:
                confirm = input("⚠️  Grids larger than 50x50 may be slow. Continue? (y/n): ")
                if confirm.lower() != 'y':
                    continue
            break
        except ValueError:
            print("❌ Please enter a valid number.\n")
    
    while True:
        try:
            cols = int(input("Enter number of columns (e.g., 10): "))
            if cols < 2:
                print("❌ Grid must have at least 2 columns. Try again.\n")
                continue
            if cols > 50:
                confirm = input("⚠️  Grids larger than 50x50 may be slow. Continue? (y/n): ")
                if confirm.lower() != 'y':
                    continue
            break
        except ValueError:
            print("❌ Please enter a valid number.\n")
    
    print(f"\n✅ Creating {rows}x{cols} grid...\n")
    return rows, cols


def choose_grid_preset():
    """Let user choose from preset grid sizes or custom"""
    print("\n" + "="*60)
    print("🎯 CHOOSE GRID SIZE")
    print("="*60)
    print("\nSelect an option:")
    print("  1. Small grid (5x5)")
    print("  2. Standard grid (10x10) [Default]")
    print("  3. Medium grid (15x15)")
    print("  4. Large grid (20x20)")
    print("  5. Extra large grid (30x30)")
    print("  6. Custom size")
    
    while True:
        try:
            choice = input("\nEnter choice (1-6) [Press Enter for default 10x10]: ").strip()
            
            if choice == "" or choice == "2":
                return 10, 10
            elif choice == "1":
                return 5, 5
            elif choice == "3":
                return 15, 15
            elif choice == "4":
                return 20, 20
            elif choice == "5":
                return 30, 30
            elif choice == "6":
                return get_grid_size()
            else:
                print("❌ Invalid choice. Please enter 1-6.\n")
        except KeyboardInterrupt:
            print("\n\nUsing default 10x10 grid...")
            return 10, 10


if __name__ == "__main__":
    try:
        # Ask user for grid size preference
        use_preset = input("Use preset grid sizes? (y/n) [Press Enter for yes]: ").strip().lower()
        
        if use_preset == 'n':
            # Custom size input
            rows, cols = get_grid_size()
        else:
            # Choose from presets
            rows, cols = choose_grid_preset()
        
        # Launch interactive grid
        print("\n🎯 Launching interactive pathfinder...")
        print(f"   Grid Size: {rows}x{cols}")
        print("   Use buttons at bottom to switch modes, then click on grid\n")
        
        interactive_grid = InteractiveGridPathfinder(rows, cols)
        interactive_grid.show()
        
    except KeyboardInterrupt:
        print("\n\n👋 Exiting program...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Launching with default 10x10 grid...")
        interactive_grid = InteractiveGridPathfinder(10, 10)
        interactive_grid.show()