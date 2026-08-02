import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, Wedge, FancyBboxPatch
from matplotlib.widgets import Slider, Button
import matplotlib.gridspec as gridspec

class MagneticEncoderSimulation:
    def __init__(self, num_magnet_pairs=6, wheel_radius=5):
        """
        Interactive magnetic encoder wheel simulation with TWO sensors
        Each magnet pair consists of one North and one South pole
        """
        self.num_magnet_pairs = num_magnet_pairs
        self.num_poles = num_magnet_pairs * 2
        self.wheel_radius = wheel_radius
        
        # Sensor 1 configuration
        self.sensor1_distance = 4.0
        self.sensor1_angle = 0
        
        # Sensor 2 configuration
        self.sensor2_distance = 4.0
        self.sensor2_angle = 90
        
        self.rotation_angle = 0
        self.angular_speed = 0
        self.time = 0
        
        # Signal history
        self.time_history = []
        self.signal1_history = []
        self.signal2_history = []
        self.max_history = 500
        
        # Setup figure with better layout
        self.fig = plt.figure(figsize=(18, 11))
        
        # Create main layout using GridSpec
        gs_main = gridspec.GridSpec(2, 2, figure=self.fig, 
                                    height_ratios=[2.5, 1], 
                                    width_ratios=[2.5, 1.5],
                                    hspace=0.35, wspace=0.3,
                                    left=0.06, right=0.98, 
                                    top=0.95, bottom=0.08)
        
        # Wheel visualization (top left)
        self.ax_wheel = self.fig.add_subplot(gs_main[0, 0])
        self.ax_wheel.set_aspect('equal')
        self.ax_wheel.set_xlim(-wheel_radius*1.6, wheel_radius*1.6)
        self.ax_wheel.set_ylim(-wheel_radius*1.6, wheel_radius*1.6)
        self.ax_wheel.grid(True, alpha=0.2, linestyle='--')
        self.ax_wheel.set_title('🧲 Magnetic Encoder Wheel with Hall Sensors', 
                               fontsize=14, fontweight='bold', pad=15)
        self.ax_wheel.set_xlabel('X Position')
        self.ax_wheel.set_ylabel('Y Position')
        
        # Signal graphs (top right) - stacked vertically
        gs_signals = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0, 1],
                                                       height_ratios=[1, 1], hspace=0.35)
        
        self.ax_signal1 = self.fig.add_subplot(gs_signals[0])
        self.ax_signal1.set_ylim(-1.3, 1.3)
        self.ax_signal1.set_xlim(0, 10)
        self.ax_signal1.grid(True, alpha=0.3)
        self.ax_signal1.set_title('Hall Sensor 1 - Output Signal', fontsize=11, fontweight='bold')
        self.ax_signal1.set_ylabel('Field Strength', fontsize=9)
        self.ax_signal1.tick_params(labelsize=8)
        
        self.ax_signal2 = self.fig.add_subplot(gs_signals[1])
        self.ax_signal2.set_ylim(-1.3, 1.3)
        self.ax_signal2.set_xlim(0, 10)
        self.ax_signal2.grid(True, alpha=0.3)
        self.ax_signal2.set_title('Hall Sensor 2 - Output Signal', fontsize=11, fontweight='bold')
        self.ax_signal2.set_xlabel('Time (seconds)', fontsize=9)
        self.ax_signal2.set_ylabel('Field Strength', fontsize=9)
        self.ax_signal2.tick_params(labelsize=8)
        
        # Controls area (bottom, spanning full width)
        gs_controls = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[1, :],
                                                        width_ratios=[1, 1, 1.2], wspace=0.35)
        
        # Sensor 1 controls
        self.ax_controls1 = self.fig.add_subplot(gs_controls[0])
        self.ax_controls1.axis('off')
        
        # Sensor 2 controls
        self.ax_controls2 = self.fig.add_subplot(gs_controls[1])
        self.ax_controls2.axis('off')
        
        # Global controls
        self.ax_global = self.fig.add_subplot(gs_controls[2])
        self.ax_global.axis('off')
        
        # Initialize all elements
        self.init_wheel()
        self.init_sensors()
        self.init_signal_plots()
        self.init_controls()
        
        # Animation
        self.anim = FuncAnimation(self.fig, self.update, frames=None, 
                                  init_func=self.init_animation,
                                  interval=50, blit=False, cache_frame_data=False)
        
    def init_wheel(self):
        """Initialize wheel with magnetic poles"""
        # Draw wheel circle
        wheel = Circle((0, 0), self.wheel_radius, fill=False, 
                      color='black', linewidth=2.5)
        self.ax_wheel.add_patch(wheel)
        
        # Draw inner circle for magnet mounting
        inner_wheel = Circle((0, 0), self.wheel_radius * 0.85, 
                            fill=False, color='gray', linewidth=1.5, 
                            linestyle='--', alpha=0.6)
        self.ax_wheel.add_patch(inner_wheel)
        
        # Draw center hub
        center = Circle((0, 0), 0.35, fill=True, color='#2C3E50')
        self.ax_wheel.add_patch(center)
        
        # Center dot
        center_dot = Circle((0, 0), 0.1, fill=True, color='white')
        self.ax_wheel.add_patch(center_dot)
        
        # Draw magnetic poles
        self.north_poles = []
        self.south_poles = []
        self.magnet_labels = []
        
        for i in range(self.num_magnet_pairs):
            north_angle = i * (360 / self.num_magnet_pairs)
            north_pole = self.create_magnet_pole(north_angle, 'N')
            self.north_poles.append(north_pole)
            
            south_angle = north_angle + (360 / self.num_poles)
            south_pole = self.create_magnet_pole(south_angle, 'S')
            self.south_poles.append(south_pole)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#E74C3C', edgecolor='#922B21', label='North Pole (N)'),
            Patch(facecolor='#3498DB', edgecolor='#1F618D', label='South Pole (S)')
        ]
        self.ax_wheel.legend(handles=legend_elements, loc='lower right', 
                            fontsize=9, framealpha=0.9)
        
    def create_magnet_pole(self, angle, pole_type):
        """Create a magnetic pole at given angle"""
        pole_angle = 360 / self.num_poles
        pole_width = pole_angle * 0.8
        
        start_angle = angle - pole_width/2
        end_angle = angle + pole_width/2
        
        if pole_type == 'N':
            color = '#E74C3C'  # Red
            edge_color = '#922B21'
        else:
            color = '#3498DB'  # Blue
            edge_color = '#1F618D'
        
        pole = Wedge(
            center=(0, 0),
            r=self.wheel_radius * 0.85,
            theta1=start_angle,
            theta2=end_angle,
            width=self.wheel_radius * 0.15,
            facecolor=color, 
            edgecolor=edge_color, 
            linewidth=1.5,
            alpha=0.75
        )
        self.ax_wheel.add_patch(pole)
        
        # Add N/S label
        label_angle_rad = np.radians(angle)
        label_radius = self.wheel_radius * 0.77
        label_x = label_radius * np.cos(label_angle_rad)
        label_y = label_radius * np.sin(label_angle_rad)
        
        label = self.ax_wheel.text(
            label_x, label_y, pole_type,
            ha='center', va='center',
            fontsize=11, fontweight='bold',
            color='white'
        )
        self.magnet_labels.append(label)
        
        return pole
    
    def init_sensors(self):
        """Initialize Hall effect sensor visual elements"""
        # Sensor 1
        sensor1_x = self.sensor1_distance * np.cos(np.radians(self.sensor1_angle))
        sensor1_y = self.sensor1_distance * np.sin(np.radians(self.sensor1_angle))
        
        # Sensor 1 body
        self.sensor1_body = FancyBboxPatch(
            (sensor1_x - 0.35, sensor1_y - 0.25),
            0.7, 0.5,
            boxstyle="round,pad=0.08",
            facecolor='#F39C12', edgecolor='#D68910', linewidth=2.5
        )
        self.ax_wheel.add_patch(self.sensor1_body)
        
        # Sensor 1 label on body
        self.ax_wheel.text(sensor1_x, sensor1_y, 'H1', 
                          ha='center', va='center', fontsize=8, 
                          fontweight='bold', color='white')
        
        self.sensor1_point, = self.ax_wheel.plot(
            [sensor1_x], [sensor1_y], 'o', color='#F1C40F', 
            markersize=14, markeredgecolor='#2C3E50', markeredgewidth=2,
            label='Hall Sensor 1', zorder=5
        )
        
        self.sensor1_label = self.ax_wheel.annotate(
            'Sensor 1', 
            xy=(sensor1_x, sensor1_y),
            xytext=(sensor1_x + 1.8, sensor1_y + 1.8),
            arrowprops=dict(arrowstyle='->', color='#F39C12', lw=2.5),
            fontsize=10, color='#D68910', weight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor='#F39C12', alpha=0.9)
        )
        
        # Sensor 2
        sensor2_x = self.sensor2_distance * np.cos(np.radians(self.sensor2_angle))
        sensor2_y = self.sensor2_distance * np.sin(np.radians(self.sensor2_angle))
        
        # Sensor 2 body
        self.sensor2_body = FancyBboxPatch(
            (sensor2_x - 0.35, sensor2_y - 0.25),
            0.7, 0.5,
            boxstyle="round,pad=0.08",
            facecolor='#9B59B6', edgecolor='#6C3483', linewidth=2.5
        )
        self.ax_wheel.add_patch(self.sensor2_body)
        
        # Sensor 2 label on body
        self.ax_wheel.text(sensor2_x, sensor2_y, 'H2', 
                          ha='center', va='center', fontsize=8, 
                          fontweight='bold', color='white')
        
        self.sensor2_point, = self.ax_wheel.plot(
            [sensor2_x], [sensor2_y], 'o', color='#BB8FCE', 
            markersize=14, markeredgecolor='#2C3E50', markeredgewidth=2,
            label='Hall Sensor 2', zorder=5
        )
        
        self.sensor2_label = self.ax_wheel.annotate(
            'Sensor 2', 
            xy=(sensor2_x, sensor2_y),
            xytext=(sensor2_x + 1.8, sensor2_y - 1.8),
            arrowprops=dict(arrowstyle='->', color='#9B59B6', lw=2.5),
            fontsize=10, color='#6C3483', weight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor='#9B59B6', alpha=0.9)
        )
        
        self.ax_wheel.legend(loc='upper right', fontsize=9, framealpha=0.9)
        
    def init_signal_plots(self):
        """Initialize signal plots"""
        # Sensor 1 signal
        self.signal1_line, = self.ax_signal1.plot([], [], color='#E74C3C', 
                                                   linewidth=2, label='Sensor 1')
        self.ax_signal1.legend(loc='upper right', fontsize=8)
        self.ax_signal1.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        self.ax_signal1.axhline(y=1, color='#E74C3C', linestyle='--', alpha=0.3, linewidth=0.8)
        self.ax_signal1.axhline(y=-1, color='#3498DB', linestyle='--', alpha=0.3, linewidth=0.8)
        self.ax_signal1.text(0.02, 0.95, 'N Pole', transform=self.ax_signal1.transAxes, 
                            fontsize=7, color='#E74C3C', va='top')
        self.ax_signal1.text(0.02, 0.05, 'S Pole', transform=self.ax_signal1.transAxes, 
                            fontsize=7, color='#3498DB', va='bottom')
        
        # Sensor 2 signal
        self.signal2_line, = self.ax_signal2.plot([], [], color='#3498DB', 
                                                   linewidth=2, label='Sensor 2')
        self.ax_signal2.legend(loc='upper right', fontsize=8)
        self.ax_signal2.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        self.ax_signal2.axhline(y=1, color='#E74C3C', linestyle='--', alpha=0.3, linewidth=0.8)
        self.ax_signal2.axhline(y=-1, color='#3498DB', linestyle='--', alpha=0.3, linewidth=0.8)
    
    def init_controls(self):
        """Initialize all control sliders and buttons with proper spacing"""
        
        # === SENSOR 1 CONTROLS (Left Panel) ===
        self.ax_controls1.text(0.5, 0.95, '🎛️ Sensor 1 Settings', 
                              transform=self.ax_controls1.transAxes,
                              ha='center', va='top', fontsize=11, 
                              fontweight='bold', color='#D68910')
        
        # S1 Angle slider
        ax_s1_angle = plt.axes([0.08, 0.35, 0.24, 0.025])
        self.s1_angle_slider = Slider(
            ax=ax_s1_angle, label='Angle (degrees)',
            valmin=0, valmax=360, valinit=self.sensor1_angle, valstep=1,
            color='#F39C12'
        )
        self.s1_angle_slider.on_changed(self.update_sensor1_angle)
        
        # S1 Distance slider
        ax_s1_distance = plt.axes([0.08, 0.28, 0.24, 0.025])
        self.s1_distance_slider = Slider(
            ax=ax_s1_distance, label='Distance from center',
            valmin=0.5, valmax=self.wheel_radius, valinit=self.sensor1_distance, valstep=0.1,
            color='#F39C12'
        )
        self.s1_distance_slider.on_changed(self.update_sensor1_distance)
        
        # === SENSOR 2 CONTROLS (Middle Panel) ===
        self.ax_controls2.text(0.5, 0.95, '🎛️ Sensor 2 Settings', 
                              transform=self.ax_controls2.transAxes,
                              ha='center', va='top', fontsize=11, 
                              fontweight='bold', color='#6C3483')
        
        # S2 Angle slider
        ax_s2_angle = plt.axes([0.37, 0.35, 0.24, 0.025])
        self.s2_angle_slider = Slider(
            ax=ax_s2_angle, label='Angle (degrees)',
            valmin=0, valmax=360, valinit=self.sensor2_angle, valstep=1,
            color='#9B59B6'
        )
        self.s2_angle_slider.on_changed(self.update_sensor2_angle)
        
        # S2 Distance slider
        ax_s2_distance = plt.axes([0.37, 0.28, 0.24, 0.025])
        self.s2_distance_slider = Slider(
            ax=ax_s2_distance, label='Distance from center',
            valmin=0.5, valmax=self.wheel_radius, valinit=self.sensor2_distance, valstep=0.1,
            color='#9B59B6'
        )
        self.s2_distance_slider.on_changed(self.update_sensor2_distance)
        
        # === GLOBAL CONTROLS (Right Panel) ===
        self.ax_global.text(0.5, 0.95, '⚙️ Global Settings', 
                           transform=self.ax_global.transAxes,
                           ha='center', va='top', fontsize=11, 
                           fontweight='bold', color='#2C3E50')
        
        # Speed slider
        ax_speed = plt.axes([0.67, 0.35, 0.26, 0.025])
        self.speed_slider = Slider(
            ax=ax_speed, label='Rotation Speed (deg/s)',
            valmin=0, valmax=360, valinit=0, valstep=1,
            color='#27AE60'
        )
        self.speed_slider.on_changed(self.update_speed)
        
        # Magnet pairs slider
        ax_magnets = plt.axes([0.67, 0.28, 0.26, 0.025])
        self.magnet_slider = Slider(
            ax=ax_magnets, label='Magnet Pole Pairs',
            valmin=1, valmax=18, valinit=self.num_magnet_pairs, valstep=1,
            color='#E67E22'
        )
        self.magnet_slider.on_changed(self.update_magnet_pairs)
        
        # Buttons row 1
        ax_reset = plt.axes([0.68, 0.20, 0.12, 0.035])
        self.reset_button = Button(ax_reset, '🔄 Reset', 
                                   color='#95A5A6', hovercolor='#7F8C8D')
        self.reset_button.on_clicked(self.reset)
        
        ax_reverse = plt.axes([0.82, 0.20, 0.12, 0.035])
        self.reverse_button = Button(ax_reverse, '↔️ Reverse', 
                                     color='#E74C3C', hovercolor='#C0392B')
        self.reverse_button.on_clicked(self.reverse_direction)
        
        # Buttons row 2
        ax_clear = plt.axes([0.68, 0.14, 0.12, 0.035])
        self.clear_button = Button(ax_clear, '🗑️ Clear', 
                                   color='#F39C12', hovercolor='#D68910')
        self.clear_button.on_clicked(self.clear_signal)
        
        ax_quad = plt.axes([0.82, 0.14, 0.12, 0.035])
        self.quad_button = Button(ax_quad, '90° Mech', 
                                  color='#3498DB', hovercolor='#2980B9')
        self.quad_button.on_clicked(self.set_quadrature)
        
        # Button row 3
        ax_90deg = plt.axes([0.68, 0.08, 0.26, 0.035])
        self.electrical_button = Button(ax_90deg, '⚡ 90° Electrical Phase', 
                                        color='#9B59B6', hovercolor='#8E44AD')
        self.electrical_button.on_clicked(self.set_electrical_90)
    
    def check_magnetic_field(self, sensor_angle, sensor_distance):
        """Check magnetic field at sensor position"""
        sensor_x = sensor_distance * np.cos(np.radians(sensor_angle))
        sensor_y = sensor_distance * np.sin(np.radians(sensor_angle))
        
        detection_angle = np.degrees(np.arctan2(sensor_y, sensor_x)) - self.rotation_angle
        detection_angle = detection_angle % 360
        
        pole_angle = 360 / self.num_poles
        normalized_angle = detection_angle % (pole_angle * 2)
        
        if normalized_angle < pole_angle:
            field_strength = np.cos(np.pi * normalized_angle / pole_angle)
            return field_strength
        else:
            field_strength = -np.cos(np.pi * (normalized_angle - pole_angle) / pole_angle)
            return field_strength
    
    def update(self, frame):
        """Animation update function"""
        dt = 0.05
        self.rotation_angle += self.angular_speed * dt
        self.rotation_angle = self.rotation_angle % 360
        self.time += dt
        
        # Update magnetic poles
        for i, (north_pole, south_pole) in enumerate(zip(self.north_poles, self.south_poles)):
            north_angle = i * (360 / self.num_magnet_pairs) + self.rotation_angle
            pole_angle = 360 / self.num_poles
            pole_width = pole_angle * 0.8
            
            north_start = north_angle - pole_width/2
            north_end = north_angle + pole_width/2
            north_pole.set_theta1(north_start)
            north_pole.set_theta2(north_end)
            
            south_angle = north_angle + pole_angle
            south_start = south_angle - pole_width/2
            south_end = south_angle + pole_width/2
            south_pole.set_theta1(south_start)
            south_pole.set_theta2(south_end)
        
        # Update magnet labels
        for idx, label in enumerate(self.magnet_labels):
            pole_type = 'N' if idx % 2 == 0 else 'S'
            pair_idx = idx // 2
            offset = 0 if pole_type == 'N' else (360 / self.num_poles)
            angle = pair_idx * (360 / self.num_magnet_pairs) + offset + self.rotation_angle
            angle_rad = np.radians(angle)
            label_radius = self.wheel_radius * 0.77
            label.set_position((label_radius * np.cos(angle_rad), 
                              label_radius * np.sin(angle_rad)))
            label.set_rotation(angle + 90)
        
        # Update Sensor 1
        s1_x = self.sensor1_distance * np.cos(np.radians(self.sensor1_angle))
        s1_y = self.sensor1_distance * np.sin(np.radians(self.sensor1_angle))
        
        self.sensor1_body.set_x(s1_x - 0.35)
        self.sensor1_body.set_y(s1_y - 0.25)
        self.sensor1_point.set_data([s1_x], [s1_y])
        self.sensor1_label.xy = (s1_x, s1_y)
        self.sensor1_label.xytext = (s1_x + 1.8, s1_y + 1.8)
        
        # Update Sensor 1 body text
        for text in self.ax_wheel.texts:
            if text.get_text() == 'H1':
                text.set_position((s1_x, s1_y))
                break
        
        # Update Sensor 2
        s2_x = self.sensor2_distance * np.cos(np.radians(self.sensor2_angle))
        s2_y = self.sensor2_distance * np.sin(np.radians(self.sensor2_angle))
        
        self.sensor2_body.set_x(s2_x - 0.35)
        self.sensor2_body.set_y(s2_y - 0.25)
        self.sensor2_point.set_data([s2_x], [s2_y])
        self.sensor2_label.xy = (s2_x, s2_y)
        self.sensor2_label.xytext = (s2_x + 1.8, s2_y - 1.8)
        
        # Update Sensor 2 body text
        for text in self.ax_wheel.texts:
            if text.get_text() == 'H2':
                text.set_position((s2_x, s2_y))
                break
        
        # Get magnetic field readings
        field1 = self.check_magnetic_field(self.sensor1_angle, self.sensor1_distance)
        field2 = self.check_magnetic_field(self.sensor2_angle, self.sensor2_distance)
        
        # Update sensor colors
        if field1 > 0.2:
            self.sensor1_point.set_color('#E74C3C')
            self.sensor1_body.set_facecolor('#F1948A')
        elif field1 < -0.2:
            self.sensor1_point.set_color('#3498DB')
            self.sensor1_body.set_facecolor('#85C1E9')
        else:
            self.sensor1_point.set_color('#95A5A6')
            self.sensor1_body.set_facecolor('#BDC3C7')
        
        if field2 > 0.2:
            self.sensor2_point.set_color('#E74C3C')
            self.sensor2_body.set_facecolor('#F1948A')
        elif field2 < -0.2:
            self.sensor2_point.set_color('#3498DB')
            self.sensor2_body.set_facecolor('#85C1E9')
        else:
            self.sensor2_point.set_color('#95A5A6')
            self.sensor2_body.set_facecolor('#BDC3C7')
        
        # Update signal history
        self.time_history.append(self.time)
        self.signal1_history.append(field1)
        self.signal2_history.append(field2)
        
        if len(self.time_history) > self.max_history:
            self.time_history = self.time_history[-self.max_history:]
            self.signal1_history = self.signal1_history[-self.max_history:]
            self.signal2_history = self.signal2_history[-self.max_history:]
        
        # Update signal plots
        if len(self.time_history) > 1:
            self.signal1_line.set_data(self.time_history, self.signal1_history)
            self.signal2_line.set_data(self.time_history, self.signal2_history)
            
            if self.time > 10:
                self.ax_signal1.set_xlim(self.time - 10, self.time)
                self.ax_signal2.set_xlim(self.time - 10, self.time)
            else:
                self.ax_signal1.set_xlim(0, max(10, self.time + 1))
                self.ax_signal2.set_xlim(0, max(10, self.time + 1))
        
        # Update title
        phase_diff = (self.sensor2_angle - self.sensor1_angle) % 360
        electrical_angle = phase_diff * self.num_magnet_pairs
        self.ax_wheel.set_title(
            f'🧲 Magnetic Encoder | Speed: {self.angular_speed:.0f}°/s | '
            f'Pairs: {self.num_magnet_pairs} | '
            f'Phase: {phase_diff:.0f}° Mech ({electrical_angle:.0f}° Elec)',
            fontsize=14, fontweight='bold', pad=15
        )
        
        return [self.sensor1_body, self.sensor1_point, 
                self.sensor2_body, self.sensor2_point] + self.north_poles + self.south_poles
    
    def init_animation(self):
        return [self.sensor1_body, self.sensor1_point, 
                self.sensor2_body, self.sensor2_point] + self.north_poles + self.south_poles
    
    def update_speed(self, val):
        self.angular_speed = val
    
    def update_sensor1_angle(self, val):
        self.sensor1_angle = val
    
    def update_sensor1_distance(self, val):
        self.sensor1_distance = val
    
    def update_sensor2_angle(self, val):
        self.sensor2_angle = val
    
    def update_sensor2_distance(self, val):
        self.sensor2_distance = val
    
    def update_magnet_pairs(self, val):
        new_pairs = int(val)
        if new_pairs != self.num_magnet_pairs:
            for pole in self.north_poles + self.south_poles:
                pole.remove()
            for label in self.magnet_labels:
                label.remove()
            
            self.num_magnet_pairs = new_pairs
            self.num_poles = new_pairs * 2
            self.north_poles = []
            self.south_poles = []
            self.magnet_labels = []
            
            for i in range(self.num_magnet_pairs):
                north_angle = i * (360 / self.num_magnet_pairs)
                north_pole = self.create_magnet_pole(north_angle, 'N')
                self.north_poles.append(north_pole)
                
                south_angle = north_angle + (360 / self.num_poles)
                south_pole = self.create_magnet_pole(south_angle, 'S')
                self.south_poles.append(south_pole)
    
    def set_quadrature(self, event):
        self.sensor1_angle = 0
        self.sensor2_angle = 90
        self.s1_angle_slider.set_val(0)
        self.s2_angle_slider.set_val(90)
    
    def set_electrical_90(self, event):
        electrical_90_mechanical = 90 / self.num_magnet_pairs
        self.sensor1_angle = 0
        self.sensor2_angle = electrical_90_mechanical
        self.s1_angle_slider.set_val(0)
        self.s2_angle_slider.set_val(electrical_90_mechanical)
    
    def reset(self, event):
        self.rotation_angle = 0
        self.time = 0
        self.angular_speed = 0
        self.speed_slider.set_val(0)
        self.time_history = []
        self.signal1_history = []
        self.signal2_history = []
        self.signal1_line.set_data([], [])
        self.signal2_line.set_data([], [])
        self.ax_signal1.set_xlim(0, 10)
        self.ax_signal2.set_xlim(0, 10)
    
    def reverse_direction(self, event):
        self.angular_speed = -self.angular_speed
        self.speed_slider.set_val(abs(self.angular_speed))
    
    def clear_signal(self, event):
        self.time_history = []
        self.signal1_history = []
        self.signal2_history = []
        self.signal1_line.set_data([], [])
        self.signal2_line.set_data([], [])
        self.time = 0
    
    def show(self):
        plt.show()

# Run simulation
if __name__ == "__main__":
    print("=" * 65)
    print("MAGNETIC ENCODER WHEEL SIMULATION")
    print("=" * 65)
    print("\n🧲 Features:")
    print("  • Magnetic North (Red) and South (Blue) poles on disk")
    print("  • Two Hall effect sensors for magnetic field detection")
    print("  • Sinusoidal magnetic field simulation")
    print("  • Adjustable number of magnet pole pairs")
    print("\n🎮 Controls:")
    print("  Sensor Controls:")
    print("    - Angle: Position Hall sensors around wheel")
    print("    - Distance: Adjust sensor gap from magnets")
    print("  Global Controls:")
    print("    - Speed: Rotation speed (0-360 deg/s)")
    print("    - Magnet Pairs: Number of N-S pole pairs (1-18)")
    print("  Buttons:")
    print("    - Reset, Reverse, Clear Signals")
    print("    - 90° Mechanical Quadrature")
    print("    - 90° Electrical Phase")
    print("\n" + "=" * 65)
    
    sim = MagneticEncoderSimulation(num_magnet_pairs=6, wheel_radius=5)
    sim.show()