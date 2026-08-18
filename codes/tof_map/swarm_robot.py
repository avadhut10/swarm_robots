"""
Swarm Robot class with odometry, ToF sensor, local mapping, Bug2 navigation, and state machine
"""

import numpy as np
from collections import deque

class Robot:
    def __init__(self, robot_id, start_pos, grid_size, resolution):
        self.robot_id = robot_id
        self.grid_size = grid_size
        self.resolution = resolution
        
        # State machine
        self.state = 'IDLE'  # IDLE, ASSIGNED, NAVIGATING, SCANNING, ZONE_COMPLETE
        self.assigned_zone = None
        self.zone_center = None
        
        # Position and orientation
        self.position = start_pos  # (x, y) in meters
        self.orientation = 0.0  # radians
        self.velocity = 0.3  # m/s
        self.angular_velocity = np.pi/4  # rad/s
        
        # Sensor models
        self.tof_range = 5.0  # meters
        self.tof_noise = 0.05  # meters std
        self.tof_angles = 72  # number of readings per sweep (every 5 degrees)
        self.odometry_noise = 0.02  # meters std
        self.imu_noise = np.deg2rad(0.5)  # radians std
        
        # Local occupancy grid (log-odds)
        self.local_grid = np.zeros((grid_size, grid_size))
        self.grid_hits = np.zeros((grid_size, grid_size))
        self.tof_hits = 0
        
        # Bug2 navigation parameters
        self.goal_position = None
        self.hit_point = None
        self.leave_point = None
        self.boundary_following = False
        self.m_line_angle = None
        self.path_history = deque(maxlen=500)
        
        # Battery simulation
        self.battery = 100.0
        self.battery_drain = 0.01
        
        # Log-odds update parameters
        self.l_occ = 0.8
        self.l_free = -0.4
        self.l_min = -5.0
        self.l_max = 5.0
        
        # Scanning state
        self.scan_angle = 0
        self.scan_complete = False
        
        print(f"Robot {robot_id} initialized at position {start_pos}")
        
    def update(self, arena, step_count):
        """Main update function for robot state machine"""
        self.battery = max(0, self.battery - self.battery_drain)
        if self.battery <= 0:
            self.state = 'IDLE'
            return
            
        arena.set_robot_position(self.robot_id, self.position)
            
        if self.state == 'IDLE':
            self._handle_idle()
        elif self.state == 'ASSIGNED':
            self._handle_assigned()
        elif self.state == 'NAVIGATING':
            self._handle_navigating(arena)
        elif self.state == 'SCANNING':
            self._handle_scanning(arena)
        elif self.state == 'ZONE_COMPLETE':
            self._handle_zone_complete()
            
        self._update_odometry()
        self.path_history.append(self.position)
        
    def _handle_idle(self):
        pass
        
    def _handle_assigned(self):
        if self.assigned_zone is not None:
            self.goal_position = self.assigned_zone['center']
            self.m_line_angle = np.arctan2(
                self.goal_position[1] - self.position[1],
                self.goal_position[0] - self.position[0]
            )
            self.state = 'NAVIGATING'
            self.boundary_following = False
            print(f"Robot {self.robot_id}: Navigating to zone at {self.goal_position}")
            
    def _handle_navigating(self, arena):
        if self.goal_position is None:
            self.state = 'IDLE'
            return
            
        dist_to_goal = np.sqrt(
            (self.position[0] - self.goal_position[0])**2 + 
            (self.position[1] - self.goal_position[1])**2
        )
        
        if dist_to_goal < 0.5:
            print(f"Robot {self.robot_id}: Reached goal, starting scan")
            self.state = 'SCANNING'
            self.boundary_following = False
            self.scan_angle = 0
            self.scan_complete = False
            return
            
        self._move_toward_goal(arena)
        
    def _move_toward_goal(self, arena):
        dx = self.goal_position[0] - self.position[0]
        dy = self.goal_position[1] - self.position[1]
        target_angle = np.arctan2(dy, dx)
        
        angle_diff = self._normalize_angle(target_angle - self.orientation)
        
        if abs(angle_diff) > 0.2:
            self.orientation += np.sign(angle_diff) * min(abs(angle_diff), self.angular_velocity * 0.1)
        else:
            step_size = self.velocity * 0.1
            new_pos = (
                self.position[0] + step_size * np.cos(self.orientation),
                self.position[1] + step_size * np.sin(self.orientation)
            )
            
            if not self._check_collision(new_pos, arena):
                self.position = new_pos
                self.boundary_following = False
            else:
                if not self.boundary_following:
                    self.hit_point = self.position
                    self.boundary_following = True
                    print(f"Robot {self.robot_id}: Hit obstacle at {self.hit_point}")
                self._follow_boundary(arena)
                
    def _follow_boundary(self, arena):
        step_size = self.velocity * 0.1
        forward_pos = (
            self.position[0] + step_size * np.cos(self.orientation),
            self.position[1] + step_size * np.sin(self.orientation)
        )
        
        if not self._check_collision(forward_pos, arena):
            self.position = forward_pos
            
            if self._can_leave_boundary():
                self.boundary_following = False
                self.leave_point = self.position
                print(f"Robot {self.robot_id}: Leaving boundary at {self.leave_point}")
            else:
                self.orientation += 0.05
        else:
            self.orientation -= 0.3
            
            new_pos = (
                self.position[0] + step_size * np.cos(self.orientation),
                self.position[1] + step_size * np.sin(self.orientation)
            )
            if not self._check_collision(new_pos, arena):
                self.position = new_pos
                
    def _can_leave_boundary(self):
        if self.hit_point is None or self.goal_position is None:
            return False
            
        current_dist = np.sqrt(
            (self.position[0] - self.goal_position[0])**2 + 
            (self.position[1] - self.goal_position[1])**2
        )
        hit_dist = np.sqrt(
            (self.hit_point[0] - self.goal_position[0])**2 + 
            (self.hit_point[1] - self.goal_position[1])**2
        )
        
        dx = self.goal_position[0] - self.hit_point[0]
        dy = self.goal_position[1] - self.hit_point[1]
        m_line_dist = abs(
            (self.position[0] - self.hit_point[0]) * dy - 
            (self.position[1] - self.hit_point[1]) * dx
        ) / np.sqrt(dx**2 + dy**2)
        
        return current_dist < hit_dist - 0.3 and m_line_dist < 0.3
        
    def _handle_scanning(self, arena):
        if self.scan_complete:
            self.state = 'ZONE_COMPLETE'
            return
            
        angles_to_scan = 36
        
        for _ in range(angles_to_scan):
            if self.scan_angle >= 2 * np.pi:
                self.scan_complete = True
                print(f"Robot {self.robot_id}: Scan complete! Total ToF hits: {self.tof_hits}")
                break
                
            distance = self._simulate_tof(self.scan_angle, arena)
            
            if distance is not None:
                self._update_grid_ray(self.scan_angle, distance)
                self.tof_hits += 1
                
            self.scan_angle += (2 * np.pi) / 72
            
    def _handle_zone_complete(self):
        if self.assigned_zone:
            self.assigned_zone['complete'] = True
            self.assigned_zone['assigned'] = None
            print(f"Robot {self.robot_id}: Zone complete!")
            self.assigned_zone = None
            self.zone_center = None
        self.state = 'IDLE'
        self.goal_position = None
        
    def _simulate_tof(self, angle, arena):
        abs_angle = self.orientation + angle
        dx = np.cos(abs_angle)
        dy = np.sin(abs_angle)
        
        max_range = min(self.tof_range, 20.0)
        for dist in np.arange(0.1, max_range, 0.05):
            check_x = self.position[0] + dist * dx
            check_y = self.position[1] + dist * dy
            
            if not (0 <= check_x < arena.size and 0 <= check_y < arena.size):
                return dist
                
            if self._check_point_collision((check_x, check_y), arena):
                noisy_dist = dist + np.random.normal(0, self.tof_noise)
                return max(0.1, noisy_dist)
                
        return self.tof_range
        
    def _update_grid_ray(self, angle, distance):
        abs_angle = self.orientation + angle
        dx = np.cos(abs_angle)
        dy = np.sin(abs_angle)
        
        robot_grid_x = int(self.position[0] / self.resolution)
        robot_grid_y = int(self.position[1] / self.resolution)
        
        steps = int(distance / self.resolution)
        for step in range(1, steps):
            grid_x = int(robot_grid_x + step * dx)
            grid_y = int(robot_grid_y + step * dy)
            
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.local_grid[grid_x, grid_y] += self.l_free
                self.local_grid[grid_x, grid_y] = np.clip(self.local_grid[grid_x, grid_y], self.l_min, self.l_max)
                self.grid_hits[grid_x, grid_y] += 1
                
        if distance < self.tof_range:
            end_grid_x = int(robot_grid_x + steps * dx)
            end_grid_y = int(robot_grid_y + steps * dy)
            
            if 0 <= end_grid_x < self.grid_size and 0 <= end_grid_y < self.grid_size:
                self.local_grid[end_grid_x, end_grid_y] += self.l_occ
                self.local_grid[end_grid_x, end_grid_y] = np.clip(self.local_grid[end_grid_x, end_grid_y], self.l_min, self.l_max)
                self.grid_hits[end_grid_x, end_grid_y] += 1
                
    def _update_odometry(self):
        if self.state in ['NAVIGATING', 'SCANNING']:
            self.position = (
                self.position[0] + np.random.normal(0, self.odometry_noise * 0.01),
                self.position[1] + np.random.normal(0, self.odometry_noise * 0.01)
            )
            self.orientation += np.random.normal(0, self.imu_noise * 0.01)
            
    def _check_collision(self, pos, arena):
        return self._check_point_collision(pos, arena, margin=0.2)
        
    def _check_point_collision(self, point, arena, margin=0.0):
        x, y = point
        
        if not (margin <= x < arena.size - margin and margin <= y < arena.size - margin):
            return True
            
        for wall in arena.walls:
            if (wall['x1'] - margin <= x <= wall['x2'] + margin and 
                wall['y1'] - margin <= y <= wall['y2'] + margin):
                return True
                
        for obstacle in arena.obstacles:
            dist = np.sqrt((x - obstacle['x'])**2 + (y - obstacle['y'])**2)
            if dist < (obstacle['radius'] + margin):
                return True
                
        return False
        
    def get_coverage(self):
        explored = np.abs(self.local_grid) > 0.5
        return np.sum(explored) / (self.grid_size * self.grid_size)
        
    @staticmethod
    def _normalize_angle(angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle