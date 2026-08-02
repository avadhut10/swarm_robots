"""
Robot class with odometry, ToF sensor, local mapping, Bug2 navigation, and state machine
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
        self.position = start_pos
        self.orientation = 0.0  # radians
        self.velocity = 0.3  # m/s
        self.angular_velocity = np.pi/4  # rad/s
        
        # Sensor models
        self.tof_range = 5.0  # meters
        self.tof_noise = 0.05  # meters std
        self.tof_angles = 360  # number of readings per sweep
        self.odometry_noise = 0.02  # meters std
        self.imu_noise = np.deg2rad(0.5)  # radians std
        
        # Local occupancy grid (log-odds)
        self.local_grid = np.zeros((grid_size, grid_size))
        self.grid_hits = np.zeros((grid_size, grid_size))  # Count of observations per cell
        self.tof_hits = 0
        
        # Bug2 navigation parameters
        self.goal_position = None
        self.hit_point = None  # Point where robot first hit obstacle
        self.leave_point = None  # Point where robot leaves obstacle
        self.boundary_following = False
        self.path_history = deque(maxlen=100)
        
        # Battery simulation
        self.battery = 100.0
        self.battery_drain = 0.01  # per step
        
        # Log-odds update parameters
        self.l_occ = 0.8  # Log-odds for occupied cell
        self.l_free = -0.4  # Log-odds for free cell
        self.l_min = -5.0  # Minimum log-odds value
        self.l_max = 5.0  # Maximum log-odds value
        
    def update(self, arena, step_count):
        """Main update function for robot state machine"""
        # Battery drain
        self.battery = max(0, self.battery - self.battery_drain)
        if self.battery <= 0:
            self.state = 'IDLE'
            return
            
        # State machine logic
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
            
        # Update dead reckoning
        self._update_odometry()
        
        # Log current position
        self.path_history.append(self.position)
        
    def _handle_idle(self):
        """Wait for zone assignment"""
        pass  # Auction system will assign zone
        
    def _handle_assigned(self):
        """Start navigating to assigned zone"""
        if self.assigned_zone is not None:
            self.goal_position = self.assigned_zone['center']
            self.state = 'NAVIGATING'
            
    def _handle_navigating(self, arena):
        """Bug2 navigation toward goal"""
        if self.goal_position is None:
            self.state = 'IDLE'
            return
            
        # Check if goal reached
        dist_to_goal = np.sqrt(
            (self.position[0] - self.goal_position[0])**2 + 
            (self.position[1] - self.goal_position[1])**2
        )
        
        if dist_to_goal < 0.5:  # Within 0.5m of goal
            self.state = 'SCANNING'
            self.boundary_following = False
            return
            
        # Bug2 logic
        if not self.boundary_following:
            # Move toward goal along m-line
            self._move_toward_goal(arena)
            
            # Check for obstacles
            if self._detect_obstacle(arena):
                self.hit_point = self.position
                self.boundary_following = True
        else:
            # Follow boundary
            self._follow_boundary(arena)
            
            # Check if we can leave boundary (reached m-line closer to goal)
            if self._can_leave_boundary():
                self.boundary_following = False
                self.leave_point = self.position
                
    def _move_toward_goal(self, arena):
        """Move robot toward goal position"""
        # Calculate direction to goal
        dx = self.goal_position[0] - self.position[0]
        dy = self.goal_position[1] - self.position[1]
        target_angle = np.arctan2(dy, dx)
        
        # Rotate toward goal
        angle_diff = self._normalize_angle(target_angle - self.orientation)
        if abs(angle_diff) > 0.1:
            self.orientation += np.sign(angle_diff) * min(abs(angle_diff), self.angular_velocity * 0.1)
        else:
            # Move forward
            new_pos = (
                self.position[0] + self.velocity * 0.1 * np.cos(self.orientation),
                self.position[1] + self.velocity * 0.1 * np.sin(self.orientation)
            )
            
            # Check collision
            if not self._check_collision(new_pos, arena):
                self.position = new_pos
                
    def _follow_boundary(self, arena):
        """Follow obstacle boundary"""
        # Simple wall-following behavior
        # Try moving forward
        forward_pos = (
            self.position[0] + self.velocity * 0.1 * np.cos(self.orientation),
            self.position[1] + self.velocity * 0.1 * np.sin(self.orientation)
        )
        
        if not self._check_collision(forward_pos, arena):
            self.position = forward_pos
            # Slightly turn away from obstacle
            self.orientation += 0.1
        else:
            # Turn toward open space
            self.orientation -= 0.2
            
    def _can_leave_boundary(self):
        """Check if robot can leave boundary following"""
        if self.hit_point is None or self.goal_position is None:
            return False
            
        # Check if we're on the m-line and closer to goal
        # Simplified check: are we closer to goal than hit point?
        current_dist = np.sqrt(
            (self.position[0] - self.goal_position[0])**2 + 
            (self.position[1] - self.goal_position[1])**2
        )
        hit_dist = np.sqrt(
            (self.hit_point[0] - self.goal_position[0])**2 + 
            (self.hit_point[1] - self.goal_position[1])**2
        )
        
        return current_dist < hit_dist - 0.5
        
    def _handle_scanning(self, arena):
        """Perform 360-degree ToF scan and update local grid"""
        # Simulate full rotation scan
        for angle in np.linspace(0, 2*np.pi, self.tof_angles):
            # Calculate ToF reading
            distance = self._simulate_tof(angle, arena)
            
            if distance is not None:
                # Update grid cells along ray
                self._update_grid_ray(angle, distance)
                self.tof_hits += 1
                
        # Mark zone as complete
        self.state = 'ZONE_COMPLETE'
        
    def _handle_zone_complete(self):
        """Mark zone as explored and return to pool"""
        if self.assigned_zone:
            self.assigned_zone['complete'] = True
            self.assigned_zone['assigned'] = None
            self.assigned_zone = None
            self.zone_center = None
        self.state = 'IDLE'
        
    def _simulate_tof(self, angle, arena):
        """Simulate Time-of-Flight sensor reading"""
        # Calculate direction vector
        dx = np.cos(self.orientation + angle)
        dy = np.sin(self.orientation + angle)
        
        # Ray casting with noise
        for dist in np.arange(0, self.tof_range, self.resolution):
            check_x = self.position[0] + dist * dx
            check_y = self.position[1] + dist * dy
            
            # Check bounds
            if not (0 <= check_x < arena.size and 0 <= check_y < arena.size):
                return dist
                
            # Check collision with walls and obstacles
            if self._check_point_collision((check_x, check_y), arena):
                # Add noise to measurement
                return dist + np.random.normal(0, self.tof_noise)
                
        return self.tof_range
        
    def _update_grid_ray(self, angle, distance):
        """Update occupancy grid cells along ToF ray using log-odds"""
        # Mark cells as free along ray
        for dist in np.arange(0, distance, self.resolution):
            x = int((self.position[0] + dist * np.cos(self.orientation + angle)) / self.resolution)
            y = int((self.position[1] + dist * np.sin(self.orientation + angle)) / self.resolution)
            
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.local_grid[x, y] += self.l_free
                self.local_grid[x, y] = np.clip(self.local_grid[x, y], self.l_min, self.l_max)
                self.grid_hits[x, y] += 1
                
        # Mark end point as occupied
        if distance < self.tof_range:
            end_x = int((self.position[0] + distance * np.cos(self.orientation + angle)) / self.resolution)
            end_y = int((self.position[1] + distance * np.sin(self.orientation + angle)) / self.resolution)
            
            if 0 <= end_x < self.grid_size and 0 <= end_y < self.grid_size:
                self.local_grid[end_x, end_y] += self.l_occ
                self.local_grid[end_x, end_y] = np.clip(self.local_grid[end_x, end_y], self.l_min, self.l_max)
                self.grid_hits[end_x, end_y] += 1
                
    def _update_odometry(self):
        """Update position with dead reckoning noise"""
        # Add noise to position based on movement
        if self.state == 'NAVIGATING' or self.state == 'SCANNING':
            self.position = (
                self.position[0] + np.random.normal(0, self.odometry_noise * 0.1),
                self.position[1] + np.random.normal(0, self.odometry_noise * 0.1)
            )
            self.orientation += np.random.normal(0, self.imu_noise * 0.1)
            
    def _detect_obstacle(self, arena):
        """Check for obstacles in front of robot"""
        # Check points ahead
        for dist in [0.1, 0.2, 0.3]:
            check_pos = (
                self.position[0] + dist * np.cos(self.orientation),
                self.position[1] + dist * np.sin(self.orientation)
            )
            if self._check_point_collision(check_pos, arena):
                return True
        return False
        
    def _check_collision(self, pos, arena):
        """Check if position collides with any obstacles"""
        return self._check_point_collision(pos, arena)
        
    def _check_point_collision(self, point, arena):
        """Check point collision with arena obstacles and walls"""
        x, y = point
        
        # Check bounds
        if not (0 <= x < arena.size and 0 <= y < arena.size):
            return True
            
        # Check walls (simple rectangular walls)
        for wall in arena.walls:
            if wall['x1'] <= x <= wall['x2'] and wall['y1'] <= y <= wall['y2']:
                return True
                
        # Check obstacles
        for obstacle in arena.obstacles:
            if np.sqrt((x - obstacle['x'])**2 + (y - obstacle['y'])**2) < obstacle['radius']:
                return True
                
        return False
        
    def get_coverage(self):
        """Calculate coverage percentage of local grid"""
        explored = np.abs(self.local_grid) > 0.5
        return np.sum(explored) / (self.grid_size * self.grid_size)
        
    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle