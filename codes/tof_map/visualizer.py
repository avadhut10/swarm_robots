"""
Pygame visualizer for multi-robot swarm simulation
"""

import pygame
import numpy as np

class Visualizer:
    def __init__(self, simulation, arena_size, grid_size):
        self.sim = simulation
        self.arena_size = arena_size
        self.grid_size = grid_size
        
        # Colors
        self.colors = {
            'background': (240, 240, 240),
            'wall': (50, 50, 50),
            'obstacle': (150, 50, 50),
            'robot': [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)],
            'path': [(200, 0, 0), (0, 200, 0), (0, 0, 200), (200, 200, 0), (200, 0, 200)],
            'grid_free': (255, 255, 255),
            'grid_occupied': (0, 0, 0),
            'grid_unknown': (128, 128, 128),
            'text': (0, 0, 0),
            'panel': (220, 220, 220),
            'button': (180, 180, 180),
            'zone': (100, 100, 255, 50),
            'selected_zone': (255, 100, 100, 100)
        }
        
        # Display setup
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Multi-Robot Swarm Mapping Simulation")
        
        # Panel dimensions
        self.global_map_rect = pygame.Rect(10, 10, 500, 500)
        self.status_panel_rect = pygame.Rect(520, 10, 300, 500)
        self.local_map_rects = []
        for i in range(3):
            x = 10 + (i * 170)
            y = 520
            self.local_map_rects.append(pygame.Rect(x, y, 160, 160))
        self.info_panel_rect = pygame.Rect(520, 520, 670, 270)
        
        # Button rects
        self.buttons = {
            'start': pygame.Rect(520, 520, 80, 30),
            'pause': pygame.Rect(610, 520, 80, 30),
            'reset': pygame.Rect(700, 520, 80, 30),
            'export': pygame.Rect(790, 520, 80, 30)
        }
        
        # Slider rects
        self.sliders = {
            'speed': pygame.Rect(520, 560, 200, 20),
            'noise': pygame.Rect(520, 590, 200, 20),
            'obstacles': pygame.Rect(520, 620, 200, 20)
        }
        
        # Font
        pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        self.clock = pygame.time.Clock()
        
    def handle_events(self):
        """Handle mouse events for buttons and sliders"""
        mouse_pos = pygame.mouse.get_pos()
        mouse_click = pygame.mouse.get_pressed()
        
        # Check button clicks
        if mouse_click[0]:
            for name, rect in self.buttons.items():
                if rect.collidepoint(mouse_pos):
                    if name == 'start':
                        self.sim.paused = False
                    elif name == 'pause':
                        self.sim.paused = True
                    elif name == 'reset':
                        self.sim.__init__(self.sim.n_robots, self.sim.arena_size, self.sim.arena.n_obstacles)
                    elif name == 'export':
                        self.sim._export_results()
                        
    def render(self):
        """Main render function"""
        self.screen.fill(self.colors['background'])
        
        # Render global map
        self._render_global_map()
        
        # Render status panel
        self._render_status_panel()
        
        # Render local maps
        self._render_local_maps()
        
        # Render info panel with buttons and sliders
        self._render_info_panel()
        
        pygame.display.flip()
        
    def _render_global_map(self):
        """Render the global merged map"""
        surface = pygame.Surface((self.global_map_rect.width, self.global_map_rect.height))
        surface.fill(self.colors['grid_unknown'])
        
        # Render merged grid
        grid = self.sim.map_merger.get_merged_grid()
        if grid is not None:
            # Scale grid to surface size
            scaled_grid = np.zeros((self.global_map_rect.height, self.global_map_rect.width))
            for i in range(self.global_map_rect.height):
                for j in range(self.global_map_rect.width):
                    grid_i = int(i * self.grid_size / self.global_map_rect.height)
                    grid_j = int(j * self.grid_size / self.global_map_rect.width)
                    if 0 <= grid_i < self.grid_size and 0 <= grid_j < self.grid_size:
                        value = grid[grid_j, grid_i]
                        # Convert log-odds to color
                        if value > 0:
                            color_val = min(255, int(128 + 127 * value / 5.0))
                            color = (color_val, color_val, color_val)
                        elif value < 0:
                            color_val = min(255, int(128 - 127 * abs(value) / 5.0))
                            color = (color_val, color_val, color_val)
                        else:
                            color = self.colors['grid_unknown']
                        surface.set_at((j, i), color)
                        
        # Draw walls
        for wall in self.sim.arena.walls:
            rect = pygame.Rect(
                wall['x1'] * self.global_map_rect.width / self.arena_size,
                wall['y1'] * self.global_map_rect.height / self.arena_size,
                (wall['x2'] - wall['x1']) * self.global_map_rect.width / self.arena_size,
                (wall['y2'] - wall['y1']) * self.global_map_rect.height / self.arena_size
            )
            pygame.draw.rect(surface, self.colors['wall'], rect)
            
        # Draw obstacles
        for obstacle in self.sim.arena.obstacles:
            pos = (
                int(obstacle['x'] * self.global_map_rect.width / self.arena_size),
                int(obstacle['y'] * self.global_map_rect.height / self.arena_size)
            )
            radius = int(obstacle['radius'] * self.global_map_rect.width / self.arena_size)
            pygame.draw.circle(surface, self.colors['obstacle'], pos, radius)
            
        # Draw zones
        for zone in self.sim.zones:
            if zone['complete']:
                color = self.colors['zone']
            elif zone['assigned'] is not None:
                color = self.colors['selected_zone']
            else:
                continue
                
            center = (
                int(zone['center'][0] * self.global_map_rect.width / self.arena_size),
                int(zone['center'][1] * self.global_map_rect.height / self.arena_size)
            )
            pygame.draw.circle(surface, color, center, 10, 2)
            
        # Draw robot paths and positions
        for robot in self.sim.robots:
            # Draw path
            path_points = list(robot.path_history)
            if len(path_points) > 1:
                scaled_path = [
                    (int(p[0] * self.global_map_rect.width / self.arena_size),
                     int(p[1] * self.global_map_rect.height / self.arena_size))
                    for p in path_points
                ]
                for i in range(len(scaled_path) - 1):
                    pygame.draw.line(surface, self.colors['path'][robot.robot_id],
                                   scaled_path[i], scaled_path[i + 1], 2)
                    
            # Draw robot
            robot_pos = (
                int(robot.position[0] * self.global_map_rect.width / self.arena_size),
                int(robot.position[1] * self.global_map_rect.height / self.arena_size)
            )
            pygame.draw.circle(surface, self.colors['robot'][robot.robot_id], robot_pos, 5)
            
            # Draw orientation line
            end_pos = (
                robot_pos[0] + int(10 * np.cos(robot.orientation)),
                robot_pos[1] + int(10 * np.sin(robot.orientation))
            )
            pygame.draw.line(surface, self.colors['robot'][robot.robot_id], robot_pos, end_pos, 2)
            
        # Draw panel border
        pygame.draw.rect(surface, (0, 0, 0), surface.get_rect(), 2)
        
        # Add title
        title = self.small_font.render("Global Merged Map", True, self.colors['text'])
        surface.blit(title, (5, 5))
        
        self.screen.blit(surface, self.global_map_rect)
        
    def _render_status_panel(self):
        """Render the status table panel"""
        surface = pygame.Surface((self.status_panel_rect.width, self.status_panel_rect.height))
        surface.fill(self.colors['panel'])
        
        # Title
        title = self.font.render("Robot Status", True, self.colors['text'])
        surface.blit(title, (10, 10))
        
        # Table header
        headers = ["ID", "State", "Zone", "Coverage", "Hits", "Battery"]
        x_positions = [10, 50, 130, 200, 270, 330]
        
        for i, header in enumerate(headers):
            text = self.small_font.render(header, True, self.colors['text'])
            surface.blit(text, (x_positions[i], 40))
            
        # Divider line
        pygame.draw.line(surface, (0, 0, 0), (10, 60), (self.status_panel_rect.width - 10, 60), 1)
        
        # Robot rows
        for i, robot in enumerate(self.sim.robots):
            y = 70 + i * 30
            
            # Robot ID
            text = self.small_font.render(str(robot.robot_id), True, self.colors['robot'][robot.robot_id])
            surface.blit(text, (x_positions[0], y))
            
            # State
            text = self.small_font.render(robot.state, True, self.colors['text'])
            surface.blit(text, (x_positions[1], y))
            
            # Zone
            zone_text = "None" if robot.assigned_zone is None else f"Zone {self.sim.zones.index(robot.assigned_zone)}"
            text = self.small_font.render(zone_text, True, self.colors['text'])
            surface.blit(text, (x_positions[2], y))
            
            # Coverage
            coverage = robot.get_coverage()
            text = self.small_font.render(f"{coverage:.1%}", True, self.colors['text'])
            surface.blit(text, (x_positions[3], y))
            
            # ToF hits
            text = self.small_font.render(str(robot.tof_hits), True, self.colors['text'])
            surface.blit(text, (x_positions[4], y))
            
            # Battery
            battery_color = (0, 255, 0) if robot.battery > 50 else (255, 255, 0) if robot.battery > 20 else (255, 0, 0)
            text = self.small_font.render(f"{robot.battery:.1f}%", True, battery_color)
            surface.blit(text, (x_positions[5], y))
            
        # Draw border
        pygame.draw.rect(surface, (0, 0, 0), surface.get_rect(), 2)
        
        self.screen.blit(surface, self.status_panel_rect)
        
    def _render_local_maps(self):
        """Render individual robot local maps"""
        for i, robot in enumerate(self.sim.robots):
            if i >= len(self.local_map_rects):
                break
                
            rect = self.local_map_rects[i]
            surface = pygame.Surface((rect.width, rect.height))
            surface.fill(self.colors['grid_unknown'])
            
            # Render local grid centered on robot
            if robot.local_grid is not None:
                # Scale and center grid around robot
                for y in range(rect.height):
                    for x in range(rect.width):
                        # Convert screen coordinates to grid coordinates
                        center_x = int(robot.position[0] / self.sim.resolution)
                        center_y = int(robot.position[1] / self.sim.resolution)
                        
                        grid_x = center_x + (x - rect.width // 2)
                        grid_y = center_y + (y - rect.height // 2)
                        
                        if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                            value = robot.local_grid[grid_x, grid_y]
                            if value > 0:
                                color_val = min(255, int(128 + 127 * value / 5.0))
                                color = (color_val, color_val, color_val)
                            elif value < 0:
                                color_val = min(255, int(128 - 127 * abs(value) / 5.0))
                                color = (color_val, color_val, color_val)
                            else:
                                color = self.colors['grid_unknown']
                            surface.set_at((x, y), color)
                            
            # Draw robot in center
            center = (rect.width // 2, rect.height // 2)
            pygame.draw.circle(surface, self.colors['robot'][robot.robot_id], center, 3)
            
            # Draw border and title
            pygame.draw.rect(surface, (0, 0, 0), surface.get_rect(), 2)
            title = self.small_font.render(f"Robot {robot.robot_id} Local", True, self.colors['text'])
            surface.blit(title, (5, 5))
            
            self.screen.blit(surface, rect)
            
    def _render_info_panel(self):
        """Render control panel with buttons, sliders, and metrics"""
        surface = pygame.Surface((self.info_panel_rect.width, self.info_panel_rect.height))
        surface.fill(self.colors['panel'])
        
        # Simulation controls
        controls_title = self.font.render("Controls", True, self.colors['text'])
        surface.blit(controls_title, (10, 10))
        
        # Buttons
        for name, rect in self.buttons.items():
            button_rect = rect.move(-self.info_panel_rect.x, -self.info_panel_rect.y)
            pygame.draw.rect(surface, self.colors['button'], button_rect)
            text = self.small_font.render(name.capitalize(), True, self.colors['text'])
            text_rect = text.get_rect(center=button_rect.center)
            surface.blit(text, text_rect)
            
        # Sliders (visual only)
        slider_y = 560 - self.info_panel_rect.y
        for name, rect in self.sliders.items():
            slider_rect = rect.move(-self.info_panel_rect.x, -self.info_panel_rect.y)
            pygame.draw.rect(surface, (200, 200, 200), slider_rect)
            pygame.draw.rect(surface, (100, 100, 100), slider_rect, 2)
            
            # Slider label
            label_text = f"{name.capitalize()}: "
            if name == 'speed':
                value = f"{self.sim.robots[0].velocity:.1f} m/s" if self.sim.robots else "0.3 m/s"
            elif name == 'noise':
                value = f"{self.sim.robots[0].tof_noise:.2f} m" if self.sim.robots else "0.05 m"
            elif name == 'obstacles':
                value = str(self.sim.arena.n_obstacles)
                
            text = self.small_font.render(label_text + value, True, self.colors['text'])
            surface.blit(text, (slider_rect.x, slider_rect.y - 15))
            
        # Metrics display
        metrics_title = self.font.render("Metrics", True, self.colors['text'])
        surface.blit(metrics_title, (400, 10))
        
        metrics = [
            f"Total Coverage: {self.sim.map_merger.get_coverage():.1%}",
            f"Steps: {self.sim.step_count}/{self.sim.max_steps}",
            f"Merge Quality: {self.sim.map_merger.quality_score:.3f}",
            f"Zone Completion: {self.sim.auctioneer.get_zone_completion():.1%}",
            f"Active Robots: {sum(1 for r in self.sim.robots if r.state != 'IDLE')}/{self.sim.n_robots}"
        ]
        
        for i, metric in enumerate(metrics):
            y = 40 + i * 25
            text = self.small_font.render(metric, True, self.colors['text'])
            surface.blit(text, (400, y))
            
        # Instructions
        instructions = [
            "Space: Pause/Resume",
            "R: Reset Simulation",
            "E: Export Map",
            "Click buttons to control"
        ]
        
        for i, instruction in enumerate(instructions):
            y = 200 + i * 20
            text = self.small_font.render(instruction, True, (100, 100, 100))
            surface.blit(text, (400, y))
            
        # Draw border
        pygame.draw.rect(surface, (0, 0, 0), surface.get_rect(), 2)
        
        self.screen.blit(surface, self.info_panel_rect)