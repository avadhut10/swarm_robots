"""
Professional UI wrapper for multi-robot swarm simulation
Uses tkinter for robust, cross-platform GUI
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import time

class SimulationUI:
    def __init__(self, simulation_engine):
        """
        Initialize UI for existing simulation engine
        
        Args:
            simulation_engine: Your existing simulation class
        """
        self.sim = simulation_engine
        self.root = tk.Tk()
        self.root.title("Multi-Robot Swarm Mapping")
        self.root.geometry("1400x900")
        
        # Simulation state
        self.running = False
        self.paused = False
        self.simulation_thread = None
        
        # Setup UI
        self._setup_styles()
        self._create_layout()
        self._create_control_panel()
        self._create_visualization_panels()
        self._create_status_panel()
        self._create_metrics_panel()
        
        # Bind keyboard shortcuts
        self.root.bind('<space>', lambda e: self.toggle_pause())
        self.root.bind('<r>', lambda e: self.reset_simulation())
        self.root.bind('<e>', lambda e: self.export_results())
        
    def _setup_styles(self):
        """Configure ttk styles for professional appearance"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', font=('Helvetica', 14, 'bold'))
        style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        style.configure('Metric.TLabel', font=('Courier', 10))
        style.configure('Status.TLabel', font=('Helvetica', 10))
        
        # Button styles
        style.configure('Start.TButton', background='#4CAF50')
        style.configure('Pause.TButton', background='#FFC107')
        style.configure('Reset.TButton', background='#F44336')
        style.configure('Export.TButton', background='#2196F3')
        
    def _create_layout(self):
        """Create main layout structure"""
        # Main container
        self.main_container = ttk.Frame(self.root, padding="10")
        self.main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_container.columnconfigure(1, weight=1)  # Visualization expands
        self.main_container.rowconfigure(1, weight=1)
        
        # Left control panel
        self.control_frame = ttk.LabelFrame(self.main_container, text="Controls", padding="10")
        self.control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Top visualization area
        self.viz_frame = ttk.Frame(self.main_container)
        self.viz_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Bottom status/metrics area
        self.status_frame = ttk.Frame(self.main_container)
        self.status_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def _create_control_panel(self):
        """Create simulation control panel"""
        # Simulation parameters
        param_frame = ttk.LabelFrame(self.control_frame, text="Parameters", padding="5")
        param_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Number of robots
        ttk.Label(param_frame, text="Robots:").grid(row=0, column=0, sticky=tk.W)
        self.robot_var = tk.IntVar(value=3)
        robot_spin = ttk.Spinbox(param_frame, from_=1, to=10, textvariable=self.robot_var, width=5)
        robot_spin.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Arena size
        ttk.Label(param_frame, text="Arena Size (m):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.size_var = tk.DoubleVar(value=20)
        size_spin = ttk.Spinbox(param_frame, from_=10, to=50, textvariable=self.size_var, width=5)
        size_spin.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Obstacles
        ttk.Label(param_frame, text="Obstacles:").grid(row=2, column=0, sticky=tk.W)
        self.obs_var = tk.IntVar(value=5)
        obs_spin = ttk.Spinbox(param_frame, from_=0, to=10, textvariable=self.obs_var, width=5)
        obs_spin.grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Sensor noise
        ttk.Label(param_frame, text="ToF Noise (m):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.noise_var = tk.DoubleVar(value=0.05)
        noise_spin = ttk.Spinbox(param_frame, from_=0.01, to=0.5, increment=0.01, 
                                textvariable=self.noise_var, width=5)
        noise_spin.grid(row=3, column=1, sticky=tk.W, padx=5)
        
        # Control buttons
        button_frame = ttk.LabelFrame(self.control_frame, text="Actions", padding="5")
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_btn = ttk.Button(button_frame, text="▶ Start", command=self.start_simulation, 
                                    style='Start.TButton')
        self.start_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.pause_btn = ttk.Button(button_frame, text="⏸ Pause", command=self.toggle_pause,
                                    style='Pause.TButton')
        self.pause_btn.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.reset_btn = ttk.Button(button_frame, text="↺ Reset", command=self.reset_simulation,
                                   style='Reset.TButton')
        self.reset_btn.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.export_btn = ttk.Button(button_frame, text="📊 Export", command=self.export_results,
                                    style='Export.TButton')
        self.export_btn.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Progress indicators
        progress_frame = ttk.LabelFrame(self.control_frame, text="Progress", padding="5")
        progress_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(progress_frame, text="Coverage:").grid(row=0, column=0, sticky=tk.W)
        self.coverage_var = tk.StringVar(value="0%")
        ttk.Label(progress_frame, textvariable=self.coverage_var, font=('Courier', 12, 'bold')).grid(
            row=0, column=1, sticky=tk.W, padx=5)
        
        self.coverage_bar = ttk.Progressbar(progress_frame, length=200, mode='determinate')
        self.coverage_bar.grid(row=1, column=0, columnspan=2, pady=5)
        
        ttk.Label(progress_frame, text="Time:").grid(row=2, column=0, sticky=tk.W)
        self.time_var = tk.StringVar(value="0 / 300")
        ttk.Label(progress_frame, textvariable=self.time_var).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Speed control
        speed_frame = ttk.LabelFrame(self.control_frame, text="Speed", padding="5")
        speed_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        self.speed_var = tk.DoubleVar(value=50)
        speed_scale = ttk.Scale(speed_frame, from_=1, to=100, variable=self.speed_var,
                               orient=tk.HORIZONTAL, length=200)
        speed_scale.grid(row=0, column=0)
        
    def _create_visualization_panels(self):
        """Create matplotlib visualization panels"""
        # Create notebook for multiple views
        self.notebook = ttk.Notebook(self.viz_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Global map tab
        self.global_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.global_frame, text="Global Map")
        
        self.global_fig = Figure(figsize=(8, 6), dpi=100)
        self.global_ax = self.global_fig.add_subplot(111)
        self.global_canvas = FigureCanvasTkAgg(self.global_fig, self.global_frame)
        self.global_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Individual robots tab
        self.robots_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.robots_frame, text="Robot Views")
        
        # Create subplots for each robot
        self.robot_figs = []
        self.robot_axes = []
        
        for i in range(3):  # Max 3 robot local views
            fig = Figure(figsize=(3, 2), dpi=80)
            ax = fig.add_subplot(111)
            canvas = FigureCanvasTkAgg(fig, self.robots_frame)
            canvas.get_tk_widget().grid(row=i//2, column=i%2, padx=5, pady=5)
            self.robot_figs.append(fig)
            self.robot_axes.append(ax)
            
    def _create_status_panel(self):
        """Create robot status display"""
        self.status_notebook = ttk.Notebook(self.status_frame)
        self.status_notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status table
        status_tab = ttk.Frame(self.status_notebook)
        self.status_notebook.add(status_tab, text="Status")
        
        # Create treeview for robot status
        columns = ('ID', 'State', 'Zone', 'Coverage', 'Hits', 'Battery')
        self.status_tree = ttk.Treeview(status_tab, columns=columns, show='headings', height=5)
        
        for col in columns:
            self.status_tree.heading(col, text=col)
            self.status_tree.column(col, width=100)
            
        self.status_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(status_tab, orient=tk.VERTICAL, command=self.status_tree.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.status_tree.configure(yscrollcommand=scrollbar.set)
        
    def _create_metrics_panel(self):
        """Create real-time metrics display"""
        metrics_tab = ttk.Frame(self.status_notebook)
        self.status_notebook.add(metrics_tab, text="Metrics")
        
        # Metrics with real-time updates
        self.metrics_text = tk.Text(metrics_tab, width=50, height=10, font=('Courier', 10))
        self.metrics_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def start_simulation(self):
        """Start simulation in separate thread"""
        if not self.running:
            self.running = True
            self.paused = False
            
            # Update button states
            self.start_btn.config(state='disabled')
            self.pause_btn.config(state='normal')
            
            # Start simulation thread
            self.simulation_thread = threading.Thread(target=self._run_simulation, daemon=True)
            self.simulation_thread.start()
            
    def toggle_pause(self):
        """Toggle simulation pause state"""
        if self.running:
            self.paused = not self.paused
            if self.paused:
                self.pause_btn.config(text="▶ Resume")
            else:
                self.pause_btn.config(text="⏸ Pause")
                
    def reset_simulation(self):
        """Reset simulation with current parameters"""
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1.0)
            
        # Reinitialize simulation with new parameters
        self.sim = self.sim.__class__(
            n_robots=self.robot_var.get(),
            arena_size=self.size_var.get(),
            n_obstacles=self.obs_var.get()
        )
        
        # Update UI
        self.update_display()
        self.start_btn.config(state='normal')
        self.pause_btn.config(text="⏸ Pause", state='disabled')
        
    def export_results(self):
        """Export simulation results"""
        if hasattr(self.sim, '_export_results'):
            self.sim._export_results()
            messagebox.showinfo("Export", "Results exported successfully!\n\n"
                               "• merged_map.npy\n"
                               "• merged_map.png")
            
    def _run_simulation(self):
        """Main simulation loop running in separate thread"""
        while self.running:
            if not self.paused:
                # Update simulation
                self.sim.update()
                
                # Update display in main thread
                self.root.after(0, self.update_display)
                
                # Check end conditions
                if self.sim.map_merger.get_coverage() >= 0.95:
                    self.root.after(0, self._simulation_complete)
                    break
                elif self.sim.step_count >= self.sim.max_steps:
                    self.root.after(0, self._simulation_complete)
                    break
                    
            # Control simulation speed
            delay = (101 - self.speed_var.get()) / 1000.0  # Convert to seconds
            time.sleep(max(delay, 0.01))
            
    def update_display(self):
        """Update all visualization elements"""
        self._update_global_map()
        self._update_robot_views()
        self._update_status_table()
        self._update_metrics()
        self._update_progress()
        
    def _update_global_map(self):
        """Update global map visualization"""
        self.global_ax.clear()
        
        if hasattr(self.sim, 'map_merger'):
            grid = self.sim.map_merger.get_merged_grid()
            
            # Create RGB visualization
            rgb_grid = np.zeros((self.sim.grid_size, self.sim.grid_size, 3))
            
            # Color mapping
            rgb_grid[grid < 0] = [1, 1, 1]  # Free space - white
            rgb_grid[grid > 0] = [0, 0, 0]  # Occupied - black
            rgb_grid[grid == 0] = [0.5, 0.5, 0.5]  # Unknown - gray
            
            self.global_ax.imshow(rgb_grid.transpose(1, 0, 2), origin='lower',
                                extent=[0, self.sim.arena_size, 0, self.sim.arena_size])
            
            # Draw robots
            for robot in self.sim.robots:
                self.global_ax.plot(robot.position[0], robot.position[1], 'o',
                                  color=f'C{robot.robot_id}', markersize=8)
                
                # Draw orientation
                dx = 0.5 * np.cos(robot.orientation)
                dy = 0.5 * np.sin(robot.orientation)
                self.global_ax.arrow(robot.position[0], robot.position[1], dx, dy,
                                   head_width=0.3, head_length=0.3)
                
            self.global_ax.set_xlim(0, self.sim.arena_size)
            self.global_ax.set_ylim(0, self.sim.arena_size)
            self.global_ax.set_aspect('equal')
            self.global_ax.grid(True, alpha=0.3)
            
        self.global_canvas.draw()
        
    def _update_robot_views(self):
        """Update individual robot views"""
        for i, robot in enumerate(self.sim.robots):
            if i >= len(self.robot_axes):
                break
                
            ax = self.robot_axes[i]
            ax.clear()
            
            # Show local grid centered on robot
            center_x = int(robot.position[0] / self.sim.resolution)
            center_y = int(robot.position[1] / self.sim.resolution)
            half_size = 20
            
            x_min = max(0, center_x - half_size)
            x_max = min(self.sim.grid_size, center_x + half_size)
            y_min = max(0, center_y - half_size)
            y_max = min(self.sim.grid_size, center_y + half_size)
            
            local_region = robot.local_grid[x_min:x_max, y_min:y_max]
            
            ax.imshow(local_region.transpose(1, 0), cmap='gray', origin='lower')
            ax.plot(local_region.shape[0]//2, local_region.shape[1]//2, 'ro', markersize=3)
            ax.set_title(f'Robot {robot.robot_id}')
            ax.set_aspect('equal')
            
            self.robot_figs[i].canvas.draw()
            
    def _update_status_table(self):
        """Update robot status table"""
        # Clear existing items
        for item in self.status_tree.get_children():
            self.status_tree.delete(item)
            
        # Add robot data
        for robot in self.sim.robots:
            values = (
                robot.robot_id,
                robot.state,
                f"Zone {self.sim.zones.index(robot.assigned_zone)}" if robot.assigned_zone else "None",
                f"{robot.get_coverage():.1%}",
                robot.tof_hits,
                f"{robot.battery:.1f}%"
            )
            self.status_tree.insert('', tk.END, values=values)
            
    def _update_metrics(self):
        """Update metrics display"""
        self.metrics_text.delete(1.0, tk.END)
        
        metrics = f"""
        SIMULATION METRICS
        ==================
        
        Total Coverage: {self.sim.map_merger.get_coverage():.1%}
        Steps: {self.sim.step_count} / {self.sim.max_steps}
        Merge Quality: {self.sim.map_merger.quality_score:.3f}
        Zone Completion: {self.sim.auctioneer.get_zone_completion():.1%}
        
        Active Robots: {sum(1 for r in self.sim.robots if r.state != 'IDLE')} / {self.sim.n_robots}
        
        Per-Robot Stats:
        """
        
        for robot in self.sim.robots:
            metrics += f"""
        Robot {robot.robot_id}:
          - State: {robot.state}
          - Coverage: {robot.get_coverage():.1%}
          - ToF Hits: {robot.tof_hits}
          - Battery: {robot.battery:.1f}%
        """
        
        self.metrics_text.insert(1.0, metrics)
        
    def _update_progress(self):
        """Update progress indicators"""
        coverage = self.sim.map_merger.get_coverage()
        self.coverage_var.set(f"{coverage:.1%}")
        self.coverage_bar['value'] = coverage * 100
        
        self.time_var.set(f"{self.sim.step_count} / {self.sim.max_steps}")
        
    def _simulation_complete(self):
        """Handle simulation completion"""
        self.running = False
        self.start_btn.config(state='normal')
        self.pause_btn.config(state='disabled')
        
        # Show completion message
        coverage = self.sim.map_merger.get_coverage()
        messagebox.showinfo("Simulation Complete", 
                          f"Simulation finished!\n\n"
                          f"Final Coverage: {coverage:.1%}\n"
                          f"Steps: {self.sim.step_count}\n\n"
                          f"Results have been exported.")
        
    def run(self):
        """Start UI main loop"""
        self.root.mainloop()

# Integration wrapper for existing simulation
def integrate_with_existing(simulation_class):
    """
    Wrapper function to integrate UI with existing simulation
    
    Usage:
        from your_simulation import YourSimulation
        from simulation_ui import integrate_with_existing
        
        integrate_with_existing(YourSimulation)()
    """
    def wrapper(*args, **kwargs):
        sim = simulation_class(*args, **kwargs)
        ui = SimulationUI(sim)
        ui.run()
    return wrapper