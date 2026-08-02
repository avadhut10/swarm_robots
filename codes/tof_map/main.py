"""
Multi-Robot Swarm Collaborative Room Mapping
Main orchestrator for simulation
"""

import numpy as np
import pygame
import argparse
from robot import Robot
from arena import Arena
from auction import Auctioneer
from map_merger import MapMerger
from visualizer import Visualizer

class Simulation:
    def __init__(self, n_robots=3, arena_size=20, n_obstacles=5):
        self.n_robots = n_robots
        self.arena_size = arena_size
        self.resolution = 0.1  # 0.1m per grid cell
        self.grid_size = int(arena_size / self.resolution)
        
        # Initialize components
        self.arena = Arena(arena_size, n_obstacles, self.resolution)
        self.auctioneer = Auctioneer(self.grid_size, n_robots)
        self.map_merger = MapMerger(self.grid_size)
        
        # Create robots with different spawn positions
        spawn_positions = [
            (2, 2), (18, 2), (2, 18), (18, 18), (10, 10),
            (5, 10), (15, 10), (10, 5), (10, 15)
        ]
        self.robots = []
        for i in range(n_robots):
            self.robots.append(Robot(
                robot_id=i,
                start_pos=spawn_positions[i],
                grid_size=self.grid_size,
                resolution=self.resolution
            ))
        
        # Simulation state
        self.step_count = 0
        self.max_steps = 300
        self.coverage_threshold = 0.95
        self.running = True
        self.paused = False
        
        # Visualization
        self.visualizer = Visualizer(self, arena_size, self.grid_size)
        
        # Initialize zone partitioning
        self._init_zones()
        
    def _init_zones(self):
        """Initialize unexplored zones as grid cells"""
        zone_size = 5  # 5x5m zones
        zones_per_dim = self.arena_size // zone_size
        
        zones = []
        for i in range(zones_per_dim):
            for j in range(zones_per_dim):
                center_x = (i + 0.5) * zone_size
                center_y = (j + 0.5) * zone_size
                zones.append({
                    'center': (center_x, center_y),
                    'assigned': None,
                    'complete': False
                })
        
        self.zones = zones
        self.auctioneer.set_zones(zones)
        
    def update(self):
        """Main update loop"""
        if self.paused or not self.running:
            return
            
        self.step_count += 1
        
        # Update dynamic obstacles every 30 steps
        if self.step_count % 30 == 0:
            self.arena.update_obstacles()
        
        # Update each robot
        for robot in self.robots:
            robot.update(self.arena, self.step_count)
            
            # Global correction from ArUco camera every 5 steps
            if self.step_count % 5 == 0:
                self._apply_global_correction(robot)
        
        # Run auction for zone assignment
        self.auctioneer.run_auction(self.robots, self.step_count)
        
        # Merge maps periodically
        if self.step_count % 10 == 0:
            self.map_merger.merge_maps([r.local_grid for r in self.robots])
        
        # Check end conditions
        self._check_end_conditions()
        
    def _apply_global_correction(self, robot):
        """Apply global position correction from overhead camera"""
        true_pos = self.arena.get_robot_position(robot.robot_id)
        if true_pos is not None:
            # Add camera noise
            noisy_pos = (
                true_pos[0] + np.random.normal(0, 0.01),
                true_pos[1] + np.random.normal(0, 0.01)
            )
            # Weighted average: 70% global, 30% dead reckoning
            robot.position = (
                0.7 * noisy_pos[0] + 0.3 * robot.position[0],
                0.7 * noisy_pos[1] + 0.3 * robot.position[1]
            )
        
    def _check_end_conditions(self):
        """Check if simulation should end"""
        coverage = self.map_merger.get_coverage()
        
        if coverage >= self.coverage_threshold:
            self.running = False
            self._export_results()
        elif self.step_count >= self.max_steps:
            self.running = False
            self._export_results()
            
    def _export_results(self):
        """Export final merged map and metrics"""
        # Export merged map
        merged_grid = self.map_merger.get_merged_grid()
        np.save('merged_map.npy', merged_grid)
        
        # Create visualization using matplotlib
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(merged_grid, cmap='gray', origin='lower')
        plt.colorbar(label='Occupancy Probability')
        plt.title(f'Final Merged Map - Coverage: {self.map_merger.get_coverage():.1%}')
        plt.savefig('merged_map.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print metrics summary
        print("\n=== SIMULATION COMPLETE ===")
        print(f"Steps elapsed: {self.step_count}")
        print(f"Final coverage: {self.map_merger.get_coverage():.1%}")
        print(f"Map merge quality score: {self.map_merger.quality_score:.3f}")
        print("\nPer-Robot Statistics:")
        for robot in self.robots:
            print(f"Robot {robot.robot_id}:")
            print(f"  Coverage: {robot.get_coverage():.1%}")
            print(f"  ToF hits: {robot.tof_hits}")
            print(f"  Battery: {robot.battery:.1f}%")
            
    def run(self):
        """Main simulation loop"""
        clock = pygame.time.Clock()
        
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.__init__(self.n_robots, self.arena_size, self.arena.n_obstacles)
                    elif event.key == pygame.K_e:
                        self._export_results()
                        
            # Check UI buttons
            self.visualizer.handle_events()
            
            # Update simulation
            self.update()
            
            # Render
            self.visualizer.render()
            
            # Control simulation speed
            clock.tick(10)  # 10 FPS
            
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--robots', type=int, default=3, help='Number of robots')
    parser.add_argument('--size', type=int, default=20, help='Arena size in meters')
    parser.add_argument('--obstacles', type=int, default=5, help='Number of obstacles')
    args = parser.parse_args()
    
    sim = Simulation(n_robots=args.robots, arena_size=args.size, n_obstacles=args.obstacles)
    sim.run()