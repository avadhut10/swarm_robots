import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import matplotlib.patches as patches
import numpy as np

class SimulationUI:
    def __init__(self, width=21.0, height=28.5):
        self.width = width
        self.height = height
        self.fig = plt.figure(figsize=(18, 10))
        self.ax = self.fig.add_axes([0.04, 0.08, 0.68, 0.88])
        
        self.ax_status = self.fig.add_axes([0.75, 0.75, 0.21, 0.21])
        self.ax_status.axis('off')
        
        self.ax_btn_calc = self.fig.add_axes([0.75, 0.62, 0.21, 0.05])
        self.ax_btn_anim = self.fig.add_axes([0.75, 0.55, 0.21, 0.05])
        self.ax_btn_reset = self.fig.add_axes([0.75, 0.48, 0.21, 0.05])
        self.ax_btn_clear = self.fig.add_axes([0.75, 0.41, 0.21, 0.05])
        self.ax_mode = self.fig.add_axes([0.75, 0.20, 0.21, 0.15])
        
        self.btn_calc = Button(self.ax_btn_calc, '🔄 Calculate Paths', color='#34495E', hovercolor='#2C3E50')
        self.btn_calc.label.set_color('white')
        
        self.btn_anim = Button(self.ax_btn_anim, '▶️ Start Animation', color='#2ECC71', hovercolor='#27AE60')
        self.btn_anim.label.set_color('white')
        
        self.btn_reset = Button(self.ax_btn_reset, '🔄 Reset All', color='#E67E22', hovercolor='#D35400')
        self.btn_reset.label.set_color('white')
        
        self.btn_clear = Button(self.ax_btn_clear, '🗑️ Clear Jobs', color='#E74C3C', hovercolor='#C0392B')
        self.btn_clear.label.set_color('white')
        
        self.radio_mode = RadioButtons(self.ax_mode, ('👁️ View Mode', '📦 Add Job Mode'), active=0, activecolor='#3498DB')
        self.robot_manager = None
        self.is_animating = False
        
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.click_callbacks = []

    def register_click_callback(self, callback):
        self.click_callbacks.append(callback)

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        if 'Add Job' in self.radio_mode.value_selected:
            for callback in self.click_callbacks:
                callback(event.xdata, event.ydata)

    def set_manager(self, manager):
        self.robot_manager = manager

    def draw(self):
        if self.robot_manager is None:
            return
        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        
        self.ax.set_xticks(np.arange(0, self.width + 0.1, 1.0))
        self.ax.set_yticks(np.arange(0, self.height + 0.1, 1.0))
        self.ax.grid(True, which='both', linestyle=':', linewidth=0.5, color='#BDC3C7')
        
        self.ax.set_facecolor('#F8F9F9')
        
        # Draw START
        start_pos = self.robot_manager.get_start()
        if start_pos:
            self.ax.scatter(start_pos[0], start_pos[1], marker='*', s=350, color='#2ECC71', edgecolors='#27AE60', zorder=5)
            self.ax.text(start_pos[0], start_pos[1] + 0.6, "★ START", color='#27AE60', fontsize=10, weight='bold', ha='center')

        # Draw END
        end_pos = self.robot_manager.get_end()
        if end_pos:
            self.ax.scatter(end_pos[0], end_pos[1], marker='*', s=350, color='#E74C3C', edgecolors='#C0392B', zorder=5)
            self.ax.text(end_pos[0], end_pos[1] + 0.6, "★ END", color='#C0392B', fontsize=10, weight='bold', ha='center')

        # Draw Jobs
        for job in self.robot_manager.get_jobs():
            if job['picked']: continue
            pos = job['pos']
            square = patches.Rectangle((pos[0] - 0.3, pos[1] - 0.3), 0.6, 0.6, linewidth=1.5, edgecolor='#2C3E50', facecolor=job['color'], zorder=4)
            self.ax.add_patch(square)
            self.ax.text(pos[0], pos[1] + 0.5, job['id'], color='#2C3E50', fontsize=9, weight='bold', ha='center')

        # Draw Robots
        for robot in self.robot_manager.get_robots():
            pos = robot['pos']
            rect = patches.Rectangle((pos[0] - 0.4, pos[1] - 0.4), 0.8, 0.8, linewidth=2.0, edgecolor='#2C3E50', facecolor=robot['color'], zorder=4)
            self.ax.add_patch(rect)
            self.ax.text(pos[0], pos[1] + 0.6, robot['id'], color='#2C3E50', fontsize=10, weight='bold', ha='center')
            
            path = robot['path']
            if len(path) > 1:
                path_x, path_y = zip(*path)
                self.ax.plot(path_x, path_y, color=robot['color'], linestyle='--', linewidth=1.5, alpha=0.8)
                
            waypoints = robot['waypoints']
            if len(waypoints) > 0:
                wp_x, wp_y = zip(*[pos] + list(waypoints))
                self.ax.plot(wp_x, wp_y, color=robot['color'], linestyle=':', linewidth=2.0, alpha=0.5)

        self.fig.canvas.draw_idle()
