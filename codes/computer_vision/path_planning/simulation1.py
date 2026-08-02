import json
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import numpy as np
import glob
import os
from datetime import datetime

class SwarmSimulator:
    """
    Swarm Robot Simulator with full workspace view
    """
    
    def __init__(self, map_file=None):
        """Initialize simulator"""
        self.robots = {}
        self.tasks = {}
        self.start_marker = None
        self.end_marker = None
        self.boundary_markers = {}
        self.workspace_width = 2800
        self.workspace_height = 2200
        self.time = 0
        
        if map_file and os.path.exists(map_file):
            print(f" Using map: {map_file}")
            self.load_map(map_file)
        else:
            # Search for map
            map_files = []
            map_files.extend(glob.glob("workspace_map_*.json"))
            map_files.extend(glob.glob("../Auruco_detection/workspace_map_*.json"))
            map_files.extend(glob.glob("F:/swarm_robots/codes/computer_vision/Auruco_detection/workspace_map_*.json"))
            
            if map_files:
                map_file = sorted(map_files)[-1]
                print(f" Found map: {map_file}")
                self.load_map(map_file)
            else:
                print(" No map file found!")
        
        print("\nRobot Swarm Simulator Initialized")
        print("="*60)
    
    def load_map(self, map_file):
        """Load workspace map"""
        try:
            map_file = map_file.replace('\\', '/')
            
            with open(map_file, 'r') as f:
                data = json.load(f)
            
            self.workspace_width = data.get('workspace_size_mm', [2800, 2200])[0]
            self.workspace_height = data.get('workspace_size_mm', [2800, 2200])[1]
            
            positions = data.get('positions', {})
            
            # Clear existing
            self.robots = {}
            self.tasks = {}
            self.start_marker = None
            self.end_marker = None
            self.boundary_markers = {}
            
            for marker_id_str, pos in positions.items():
                marker_id = int(marker_id_str)
                label = pos.get('label', f'ID{marker_id}')
                x = pos.get('x_mm', 0)
                y = pos.get('y_mm', 0)
                
                if marker_id in [100, 101, 102]:
                    # Robot
                    self.robots[marker_id] = {
                        'id': marker_id,
                        'x': x,
                        'y': y,
                        'start_x': x,
                        'start_y': y,
                        'label': label,
                        'target_x': x,
                        'target_y': y,
                        'speed': 100,
                        'state': 'idle',
                        'history': []
                    }
                elif marker_id == 10:
                    self.start_marker = {'x': x, 'y': y, 'label': 'START'}
                elif marker_id == 11:
                    self.end_marker = {'x': x, 'y': y, 'label': 'END'}
                elif marker_id in [20, 21]:
                    self.tasks[marker_id] = {
                        'id': marker_id,
                        'x': x,
                        'y': y,
                        'label': label,
                        'status': 'pending',
                        'assigned_to': None
                    }
                elif marker_id in [0, 1, 2, 3]:
                    self.boundary_markers[marker_id] = {'x': x, 'y': y, 'label': label}
            
            print(f" Map loaded successfully!")
            print(f"   Workspace: {self.workspace_width}x{self.workspace_height}mm")
            print(f"   Robots: {len(self.robots)}")
            print(f"   Tasks: {len(self.tasks)}")
            print(f"   Start: {'Yes' if self.start_marker else 'No'}")
            print(f"   End: {'Yes' if self.end_marker else 'No'}")
            
            return True
            
        except Exception as e:
            print(f" Could not load map: {e}")
            return False
    
    def assign_tasks(self):
        """Assign tasks to nearest robots"""
        print("\nAssigning tasks to robots...")
        print("-"*40)
        
        for task in self.tasks.values():
            task['status'] = 'pending'
            task['assigned_to'] = None
        
        for robot in self.robots.values():
            robot['state'] = 'idle'
            robot['history'] = []
        
        for task_id, task in self.tasks.items():
            if task['status'] != 'pending':
                continue
            
            nearest_robot = None
            nearest_distance = float('inf')
            
            for robot_id, robot in self.robots.items():
                assigned = False
                for t in self.tasks.values():
                    if t['assigned_to'] == robot_id and t['status'] != 'complete':
                        assigned = True
                        break
                
                if assigned:
                    continue
                
                dist = math.sqrt((task['x'] - robot['x'])**2 + 
                               (task['y'] - robot['y'])**2)
                
                if dist < nearest_distance:
                    nearest_distance = dist
                    nearest_robot = robot_id
            
            if nearest_robot is not None:
                task['assigned_to'] = nearest_robot
                task['status'] = 'assigned'
                
                self.robots[nearest_robot]['target_x'] = task['x']
                self.robots[nearest_robot]['target_y'] = task['y']
                self.robots[nearest_robot]['state'] = 'moving'
                
                print(f"   {task['label']} -> Robot {nearest_robot-99}")
                print(f"      Distance: {nearest_distance:.0f}mm")
            else:
                print(f"   {task['label']} -> No robot available!")
        
        print("-"*40)
    
    def update(self, dt=0.1, steps=100):
        """Run simulation"""
        print(f"\nRunning simulation for {steps} steps...")
        
        for step in range(steps):
            self.time += dt
            
            for robot_id, robot in self.robots.items():
                if robot['state'] != 'moving':
                    continue
                
                dx = robot['target_x'] - robot['x']
                dy = robot['target_y'] - robot['y']
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance < 1:
                    robot['x'] = robot['target_x']
                    robot['y'] = robot['target_y']
                    robot['state'] = 'idle'
                    
                    for task_id, task in self.tasks.items():
                        if task['assigned_to'] == robot_id:
                            task['status'] = 'complete'
                            print(f"   [DONE] Robot {robot_id-99} completed {task['label']}")
                else:
                    speed = robot['speed'] * dt
                    step_size = min(speed, distance)
                    robot['x'] += (dx / distance) * step_size
                    robot['y'] += (dy / distance) * step_size
                    robot['history'].append((robot['x'], robot['y']))
                    
                    if len(robot['history']) > 50:
                        robot['history'].pop(0)
            
            all_complete = all(t['status'] == 'complete' for t in self.tasks.values())
            if all_complete and self.tasks:
                print(f"\n   All tasks complete at time: {self.time:.1f}s")
                break
            
            if step % 20 == 0 and step > 0:
                complete = sum(1 for t in self.tasks.values() if t['status'] == 'complete')
                total = len(self.tasks)
                print(f"   Time: {self.time:.1f}s - Tasks: {complete}/{total}")
    
    def visualize(self):
        """Create FULLY VISIBLE visualization with auto-zoom"""
        if not self.robots and not self.tasks:
            print(" No data to visualize!")
            return
        
        # Create figure with large size
        fig, ax = plt.subplots(figsize=(18, 12), dpi=100)
        
        # Calculate workspace bounds with padding
        padding = 200  # Extra space around workspace
        x_min = -padding
        x_max = self.workspace_width + padding
        y_min = -padding
        y_max = self.workspace_height + padding
        
        # Draw workspace background
        ax.fill_between([0, self.workspace_width], 0, self.workspace_height, 
                        alpha=0.08, color='blue')
        
        # Draw workspace border
        rect = Rectangle((0, 0), self.workspace_width, self.workspace_height, 
                        fill=False, edgecolor='black', linewidth=3)
        ax.add_patch(rect)
        
        # Draw grid (500mm spacing)
        for x in range(0, self.workspace_width + 1, 500):
            ax.axvline(x, color='gray', alpha=0.3, linestyle='--', linewidth=0.5)
            if x < self.workspace_width:
                ax.text(x, -35, f'{x}', fontsize=9, ha='center', va='top', color='gray')
        
        for y in range(0, self.workspace_height + 1, 500):
            ax.axhline(y, color='gray', alpha=0.3, linestyle='--', linewidth=0.5)
            if y < self.workspace_height:
                ax.text(-35, y, f'{y}', fontsize=9, ha='right', va='center', color='gray')
        
        # Draw Boundary markers (IDs 0-3)
        for marker_id, pos in self.boundary_markers.items():
            ax.scatter(pos['x'], pos['y'], s=120, color='green', 
                      marker='s', zorder=2, alpha=0.8, edgecolor='darkgreen', linewidth=2)
            ax.text(pos['x'], pos['y'] + 25, f'B{marker_id}', 
                   fontsize=9, ha='center', color='darkgreen', fontweight='bold')
            ax.text(pos['x'], pos['y'] - 25, f"({pos['x']:.0f},{pos['y']:.0f})", 
                   fontsize=7, ha='center', color='gray')
        
        # Draw Start marker (ID 10)
        if self.start_marker:
            ax.scatter(self.start_marker['x'], self.start_marker['y'], 
                      s=300, color='lime', marker='*', zorder=3, 
                      edgecolor='darkgreen', linewidth=2)
            ax.text(self.start_marker['x'], self.start_marker['y'] + 40, 
                   '🏁 START', fontsize=12, ha='center', fontweight='bold', color='darkgreen')
            ax.text(self.start_marker['x'], self.start_marker['y'] - 35, 
                   f"({self.start_marker['x']:.0f}, {self.start_marker['y']:.0f})", 
                   fontsize=8, ha='center', color='gray')
        
        # Draw End marker (ID 11)
        if self.end_marker:
            ax.scatter(self.end_marker['x'], self.end_marker['y'], 
                      s=300, color='red', marker='*', zorder=3, 
                      edgecolor='darkred', linewidth=2)
            ax.text(self.end_marker['x'], self.end_marker['y'] + 40, 
                   '🏁 END', fontsize=12, ha='center', fontweight='bold', color='darkred')
            ax.text(self.end_marker['x'], self.end_marker['y'] - 35, 
                   f"({self.end_marker['x']:.0f}, {self.end_marker['y']:.0f})", 
                   fontsize=8, ha='center', color='gray')
        
        # Draw Tasks (Jobs - IDs 20, 21)
        job_colors = {'JOB 1': '#FF6B6B', 'JOB 2': '#4ECDC4'}
        
        for task_id, task in self.tasks.items():
            if task['status'] == 'complete':
                color = '#2ECC71'  # Green
                edgecolor = '#27AE60'
                marker = 's'
                status_text = '✓ COMPLETE'
            elif task['status'] == 'assigned':
                color = '#F39C12'  # Orange
                edgecolor = '#E67E22'
                marker = 's'
                status_text = '⏳ IN PROGRESS'
            else:
                color = '#E74C3C'  # Red
                edgecolor = '#C0392B'
                marker = 's'
                status_text = '✗ PENDING'
            
            # Task marker - large square
            ax.scatter(task['x'], task['y'], s=500, color=color, 
                      marker=marker, zorder=3, alpha=0.8, 
                      edgecolor=edgecolor, linewidth=3)
            
            # Task label at top
            ax.text(task['x'], task['y'] + 50, task['label'], 
                   fontsize=12, ha='center', va='bottom', fontweight='bold',
                   color='black', 
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                           edgecolor=color, alpha=0.95))
            
            # Status text
            ax.text(task['x'], task['y'] - 10, status_text, 
                   fontsize=9, ha='center', va='center', 
                   color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=edgecolor, alpha=0.9))
            
            # Coordinates
            ax.text(task['x'], task['y'] - 45, f"({task['x']:.0f}, {task['y']:.0f})", 
                   fontsize=8, ha='center', va='top', color='gray')
            
            # Assigned robot info
            if task['assigned_to']:
                ax.text(task['x'] + 70, task['y'] - 10, f'→ R{task["assigned_to"]-99}', 
                       fontsize=9, ha='left', va='center', color='blue', fontweight='bold')
        
        # Draw Robots (IDs 100, 101, 102)
        for robot_id, robot in self.robots.items():
            if robot['state'] == 'moving':
                color = '#E74C3C'  # Red
                edgecolor = '#C0392B'
                status = '▶ MOVING'
                size = 45
            else:
                color = '#3498DB'  # Blue
                edgecolor = '#2980B9'
                status = '● IDLE'
                size = 40
            
            # Robot circle with glow effect
            # Outer glow
            glow = Circle((robot['x'], robot['y']), size * 1.8, 
                         facecolor=color, alpha=0.15, zorder=3)
            ax.add_patch(glow)
            
            # Main circle
            circle = Circle((robot['x'], robot['y']), size, 
                          facecolor=color, edgecolor=edgecolor, 
                          linewidth=3, zorder=4, alpha=0.9)
            ax.add_patch(circle)
            
            # Robot ID inside circle
            ax.text(robot['x'], robot['y'], f'R{robot_id-99}', 
                   fontsize=14, ha='center', va='center', 
                   fontweight='bold', color='white')
            
            # Robot label above
            ax.text(robot['x'], robot['y'] + size + 25, status, 
                   fontsize=10, ha='center', va='bottom', 
                   color=edgecolor, fontweight='bold')
            
            # Coordinates below
            ax.text(robot['x'], robot['y'] - size - 20, 
                   f"({robot['x']:.0f}, {robot['y']:.0f})", 
                   fontsize=8, ha='center', va='top', color='gray')
            
            # Draw path history with arrow
            if len(robot['history']) > 1:
                hist = np.array(robot['history'])
                ax.plot(hist[:, 0], hist[:, 1], color=color, 
                       alpha=0.4, linewidth=2.5, linestyle='--')
                
                # Start point
                if len(hist) > 0:
                    ax.scatter(hist[0, 0], hist[0, 1], s=60, 
                             color=color, alpha=0.5, marker='o', 
                             edgecolor='black', linewidth=1)
                    ax.text(hist[0, 0], hist[0, 1] - 25, 'START', 
                           fontsize=7, ha='center', color='gray')
                
                # Target point (if different from current)
                if robot['state'] == 'moving':
                    target_x = robot['target_x']
                    target_y = robot['target_y']
                    ax.scatter(target_x, target_y, s=80, 
                             color='red', alpha=0.3, marker='x', 
                             linewidth=2)
        
        # Add workspace info box
        info_text = (
            f"WORKSPACE: {self.workspace_width} x {self.workspace_height} mm\n"
            f"ROBOTS: {len(self.robots)} | TASKS: {len(self.tasks)}\n"
            f"TIME: {self.time:.1f}s"
        )
        ax.text(x_max - 200, y_max - 50, info_text, 
               fontsize=10, ha='right', va='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                       edgecolor='black', alpha=0.9))
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ECC71', edgecolor='#27AE60', label='✓ Task Complete'),
            Patch(facecolor='#F39C12', edgecolor='#E67E22', label='⏳ Task In Progress'),
            Patch(facecolor='#E74C3C', edgecolor='#C0392B', label='✗ Task Pending'),
            Patch(facecolor='#3498DB', edgecolor='#2980B9', label='● Robot Idle'),
            Patch(facecolor='#E74C3C', edgecolor='#C0392B', label='▶ Robot Moving'),
            Patch(facecolor='lime', edgecolor='darkgreen', label='★ START'),
            Patch(facecolor='red', edgecolor='darkred', label='★ END'),
            Patch(facecolor='green', edgecolor='darkgreen', label='▣ Boundary')
        ]
        ax.legend(handles=legend_elements, loc='upper left', 
                 fontsize=10, framealpha=0.95)
        
        # Labels and title
        ax.set_xlabel('X Position (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position (mm)', fontsize=12, fontweight='bold')
        ax.set_title('🤖 SWARM ROBOT SIMULATION - FULL WORKSPACE VIEW', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Set limits to show ENTIRE workspace
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Equal aspect ratio
        ax.set_aspect('equal')
        
        # Grid
        ax.grid(True, alpha=0.1)
        
        # Make sure everything is visible
        plt.tight_layout()
        
        # Show plot with zoom controls
        print("\n" + "="*60)
        print("📊 VISUALIZATION CONTROLS:")
        print("   - Use MOUSE WHEEL to zoom in/out")
        print("   - Click and DRAG to pan")
        print("   - Click HOME button to reset view")
        print("   - Click ZOOM button then drag to zoom into specific area")
        print("   - Close the plot window to continue")
        print("="*60)
        
        plt.show()
    
    def run(self):
        """Run full simulation"""
        if not self.robots or not self.tasks:
            print("\n No robots or tasks found!")
            print("   Make sure your map has:")
            print("   - Robots: IDs 100, 101, 102")
            print("   - Tasks: IDs 20, 21")
            print("   - Start: ID 10")
            print("   - End: ID 11")
            return
        
        print("\n" + "="*60)
        print("STARTING SIMULATION")
        print("="*60)
        print(f"Workspace: {self.workspace_width}x{self.workspace_height}mm")
        print(f"Robots: {len(self.robots)}")
        print(f"Tasks: {len(self.tasks)}")
        if self.start_marker:
            print(f"Start: ({self.start_marker['x']:.0f}, {self.start_marker['y']:.0f})mm")
        if self.end_marker:
            print(f"End: ({self.end_marker['x']:.0f}, {self.end_marker['y']:.0f})mm")
        print("="*60)
        
        # Show initial positions
        print("\n📍 INITIAL POSITIONS:")
        for robot_id, robot in self.robots.items():
            print(f"   Robot {robot_id-99}: ({robot['x']:.0f}, {robot['y']:.0f})mm")
        for task_id, task in self.tasks.items():
            print(f"   {task['label']}: ({task['x']:.0f}, {task['y']:.0f})mm")
        
        # Assign tasks
        self.assign_tasks()
        
        # Run simulation
        self.update(dt=0.1, steps=100)
        
        # Show results
        print("\n" + "="*60)
        print("📊 SIMULATION COMPLETE")
        print("="*60)
        
        complete = sum(1 for t in self.tasks.values() if t['status'] == 'complete')
        total = len(self.tasks)
        print(f"Tasks completed: {complete}/{total}")
        print(f"Simulation time: {self.time:.1f}s")
        
        print("\n📍 FINAL POSITIONS:")
        for robot_id, robot in self.robots.items():
            dist = math.sqrt((robot['x'] - robot['target_x'])**2 + 
                           (robot['y'] - robot['target_y'])**2)
            status = "✅ DONE" if dist < 1 else "⏳ MOVING"
            print(f"   Robot {robot_id-99}: ({robot['x']:.0f}, {robot['y']:.0f})mm {status}")
        
        print("\n📋 TASK STATUS:")
        for task_id, task in self.tasks.items():
            status = "✅" if task['status'] == 'complete' else "⏳"
            robot_info = f"Robot {task['assigned_to']-99}" if task['assigned_to'] else "Unassigned"
            print(f"   {status} {task['label']}: {task['status']} (→ {robot_info})")
        
        # Visualize
        print("\n🎨 Generating visualization...")
        self.visualize()
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save simulation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_results_{timestamp}.json"
        
        results = {
            'timestamp': timestamp,
            'workspace': [self.workspace_width, self.workspace_height],
            'time': self.time,
            'robots': {},
            'tasks': {},
            'start': self.start_marker,
            'end': self.end_marker
        }
        
        for robot_id, robot in self.robots.items():
            results['robots'][str(robot_id)] = {
                'start': [robot['start_x'], robot['start_y']],
                'end': [robot['x'], robot['y']],
                'target': [robot['target_x'], robot['target_y']],
                'state': robot['state']
            }
        
        for task_id, task in self.tasks.items():
            results['tasks'][str(task_id)] = {
                'label': task['label'],
                'position': [task['x'], task['y']],
                'status': task['status'],
                'assigned_to': task['assigned_to']
            }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f" Results saved to: {filename}")


# Main execution
if __name__ == "__main__":
    print("Swarm Robot Simulator")
    print("="*60)
    
    # Path to your specific map file
    map_file = "F:/swarm_robots/codes/computer_vision/Auruco_detection/workspace_map_20260703_001605.json"
    
    # Create and run simulator
    simulator = SwarmSimulator(map_file)
    simulator.run()