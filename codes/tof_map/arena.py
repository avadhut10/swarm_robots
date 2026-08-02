"""
Arena class with static walls and dynamic obstacles
"""

import numpy as np

class Arena:
    def __init__(self, size, n_obstacles, resolution):
        self.size = size
        self.resolution = resolution
        self.grid_size = int(size / resolution)
        
        # Initialize walls (bounding box + internal walls)
        self.walls = self._create_walls()
        
        # Initialize obstacles
        self.n_obstacles = n_obstacles
        self.obstacles = []
        self._create_obstacles()
        
        # Ground truth grid
        self.ground_truth = self._create_ground_truth()
        
        # Robot positions for ground truth simulation
        self.robot_positions = {}
        
    def _create_walls(self):
        """Create arena boundary walls and internal walls"""
        walls = []
        
        # Boundary walls
        walls.append({'x1': 0, 'y1': 0, 'x2': self.size, 'y2': 0.2})  # Bottom
        walls.append({'x1': 0, 'y1': self.size - 0.2, 'x2': self.size, 'y2': self.size})  # Top
        walls.append({'x1': 0, 'y1': 0, 'x2': 0.2, 'y2': self.size})  # Left
        walls.append({'x1': self.size - 0.2, 'y1': 0, 'x2': self.size, 'y2': self.size})  # Right
        
        # Internal walls (example configuration)
        walls.append({'x1': 8, 'y1': 5, 'x2': 8.2, 'y2': 12})  # Vertical wall
        walls.append({'x1': 12, 'y1': 10, 'x2': 18, 'y2': 10.2})  # Horizontal wall
        
        return walls
        
    def _create_obstacles(self):
        """Create random obstacles"""
        np.random.seed(42)  # For reproducibility
        
        for _ in range(self.n_obstacles):
            while True:
                x = np.random.uniform(2, self.size - 2)
                y = np.random.uniform(2, self.size - 2)
                radius = np.random.uniform(0.3, 0.8)
                
                # Check if obstacle overlaps with walls or other obstacles
                valid = True
                
                # Check wall overlap
                for wall in self.walls:
                    if (wall['x1'] - radius < x < wall['x2'] + radius and 
                        wall['y1'] - radius < y < wall['y2'] + radius):
                        valid = False
                        break
                        
                # Check other obstacle overlap
                for obs in self.obstacles:
                    dist = np.sqrt((x - obs['x'])**2 + (y - obs['y'])**2)
                    if dist < (radius + obs['radius'] + 0.5):
                        valid = False
                        break
                        
                if valid:
                    self.obstacles.append({
                        'x': x,
                        'y': y,
                        'radius': radius,
                        'vx': np.random.uniform(-0.1, 0.1),
                        'vy': np.random.uniform(-0.1, 0.1)
                    })
                    break
                    
    def _create_ground_truth(self):
        """Create ground truth occupancy grid"""
        grid = np.zeros((self.grid_size, self.grid_size))
        
        # Mark walls as occupied
        for wall in self.walls:
            x1 = int(wall['x1'] / self.resolution)
            x2 = int(wall['x2'] / self.resolution)
            y1 = int(wall['y1'] / self.resolution)
            y2 = int(wall['y2'] / self.resolution)
            grid[x1:x2, y1:y2] = 1
            
        # Mark obstacles as occupied
        for obs in self.obstacles:
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    x = i * self.resolution
                    y = j * self.resolution
                    if np.sqrt((x - obs['x'])**2 + (y - obs['y'])**2) < obs['radius']:
                        grid[i, j] = 1
                        
        return grid
        
    def update_obstacles(self):
        """Update dynamic obstacle positions"""
        for obstacle in self.obstacles:
            # Random walk
            obstacle['vx'] += np.random.normal(0, 0.05)
            obstacle['vy'] += np.random.normal(0, 0.05)
            
            # Limit velocity
            speed = np.sqrt(obstacle['vx']**2 + obstacle['vy']**2)
            if speed > 0.2:
                obstacle['vx'] *= 0.2 / speed
                obstacle['vy'] *= 0.2 / speed
                
            # Update position
            new_x = obstacle['x'] + obstacle['vx']
            new_y = obstacle['y'] + obstacle['vy']
            
            # Bounce off walls
            if new_x - obstacle['radius'] < 0 or new_x + obstacle['radius'] > self.size:
                obstacle['vx'] *= -1
                new_x = obstacle['x']
            if new_y - obstacle['radius'] < 0 or new_y + obstacle['radius'] > self.size:
                obstacle['vy'] *= -1
                new_y = obstacle['y']
                
            obstacle['x'] = new_x
            obstacle['y'] = new_y
            
    def set_robot_position(self, robot_id, position):
        """Set ground truth robot position"""
        self.robot_positions[robot_id] = position
        
    def get_robot_position(self, robot_id):
        """Get ground truth robot position"""
        return self.robot_positions.get(robot_id)
        
    def update_robot_position(self, robot):
        """Update ground truth robot position from robot object"""
        self.robot_positions[robot.robot_id] = robot.position