"""
Multi-Robot Swarm Collaborative Room Mapping
Main orchestrator for simulation - Fixed Version
"""

import numpy as np
import argparse
from swarm_robot import Robot  # Changed import
from arena import Arena
from auction import Auctioneer
from map_merger import MapMerger

class Simulation:
    def __init__(self, n_robots=3, arena_size=20, n_obstacles=5):
        self.n_robots = n_robots
        self.arena_size = arena_size
        self.resolution = 0.1
        self.grid_size = int(arena_size / self.resolution)
        
        # Initialize components
        self.arena = Arena(arena_size, n_obstacles, self.resolution)
        self.auctioneer = Auctioneer(self.grid_size, n_robots)
        self.map_merger = MapMerger(self.grid_size)
        
        # Create robots with different spawn positions
        spawn_positions = [
            (2, 2),    # Robot 0
            (18, 2),   # Robot 1
            (2, 18),   # Robot 2
            (18, 18),  # Robot 3
            (10, 10),  # Robot 4
            (5, 10),   # Robot 5
            (15, 10),  # Robot 6
            (10, 5),   # Robot 7
            (10, 15)   # Robot 8
        ]
        
        self.robots = []
        for i in range(n_robots):
            pos = spawn_positions[i]
            robot = Robot(
                robot_id=i,
                start_pos=pos,
                grid_size=self.grid_size,
                resolution=self.resolution
            )
            self.robots.append(robot)
            self.arena.set_robot_position(i, pos)
            
        print(f"\n=== Initialized {n_robots} robots ===")
        for robot in self.robots:
            print(f"Robot {robot.robot_id}: Position {robot.position}")
        
        # Simulation state
        self.step_count = 0
        self.max_steps = 300
        self.coverage_threshold = 0.95
        self.running = True
        self.paused = False
        
        # Initialize zones
        self._init_zones()
        self.metrics_history = []
        
    def _init_zones(self):
        zone_size = 5
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
        print(f"Created {len(zones)} zones for exploration")
        
    def update(self):
        if self.paused or not self.running:
            return False
            
        self.step_count += 1
        
        if self.step_count % 30 == 0:
            self.arena.update_obstacles()
            print(f"Step {self.step_count}: Updated obstacles")
        
        # Update each robot
        for robot in self.robots:
            robot.update(self.arena, self.step_count)
            
            if self.step_count % 5 == 0:
                self._apply_global_correction(robot)
        
        # Run auction
        self.auctioneer.run_auction(self.robots, self.step_count)
        
        # Merge maps periodically
        if self.step_count % 10 == 0:
            self.map_merger.merge_maps([r.local_grid for r in self.robots])
            
            coverage = self.map_merger.get_coverage()
            print(f"Step {self.step_count}: Total Coverage: {coverage:.1%}")
            for robot in self.robots:
                print(f"  Robot {robot.robot_id}: State={robot.state}, Coverage={robot.get_coverage():.1%}, Hits={robot.tof_hits}")
        
        self.metrics_history.append({
            'step': self.step_count,
            'coverage': self.map_merger.get_coverage(),
            'quality': self.map_merger.quality_score,
            'robot_coverages': [r.get_coverage() for r in self.robots]
        })
        
        return self._check_end_conditions()
        
    def _apply_global_correction(self, robot):
        true_pos = self.arena.get_robot_position(robot.robot_id)
        if true_pos is not None:
            noisy_pos = (
                true_pos[0] + np.random.normal(0, 0.01),
                true_pos[1] + np.random.normal(0, 0.01)
            )
            robot.position = (
                0.7 * noisy_pos[0] + 0.3 * robot.position[0],
                0.7 * noisy_pos[1] + 0.3 * robot.position[1]
            )
    
    def _check_end_conditions(self):
        coverage = self.map_merger.get_coverage()
        
        if coverage >= self.coverage_threshold:
            print(f"\n=== Coverage target reached: {coverage:.1%} ===")
            self.running = False
            self._export_results()
            return True
        elif self.step_count >= self.max_steps:
            print(f"\n=== Max steps reached: {self.step_count} ===")
            self.running = False
            self._export_results()
            return True
        
        return False
            
    def _export_results(self):
        merged_grid = self.map_merger.get_merged_grid()
        np.save('merged_map.npy', merged_grid)
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(merged_grid, cmap='gray', origin='lower')
        plt.colorbar(label='Occupancy Probability')
        plt.title(f'Final Merged Map\nCoverage: {self.map_merger.get_coverage():.1%}')
        
        for i, robot in enumerate(self.robots):
            plt.subplot(2, 2, i+2)
            plt.imshow(robot.local_grid, cmap='gray', origin='lower')
            plt.title(f'Robot {robot.robot_id} Local Map\nHits: {robot.tof_hits}')
            
        plt.tight_layout()
        plt.savefig('merged_map.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\n=== SIMULATION COMPLETE ===")
        print(f"Steps elapsed: {self.step_count}")
        print(f"Final coverage: {self.map_merger.get_coverage():.1%}")
        print(f"Map merge quality score: {self.map_merger.quality_score:.3f}")
        print("\nPer-Robot Statistics:")
        for robot in self.robots:
            print(f"Robot {robot.robot_id}:")
            print(f"  State: {robot.state}")
            print(f"  Position: ({robot.position[0]:.2f}, {robot.position[1]:.2f})")
            print(f"  Coverage: {robot.get_coverage():.1%}")
            print(f"  ToF hits: {robot.tof_hits}")
            print(f"  Battery: {robot.battery:.1f}%")
            print(f"  Grid non-zero cells: {np.count_nonzero(robot.local_grid)}")
    
    def get_simulation_state(self):
        return {
            'step_count': self.step_count,
            'max_steps': self.max_steps,
            'coverage': self.map_merger.get_coverage(),
            'quality': self.map_merger.quality_score,
            'zone_completion': self.auctioneer.get_zone_completion(),
            'merged_grid': self.map_merger.get_merged_grid(),
            'robots': self.get_robot_data(),
            'zones': self.zones,
            'arena': {
                'size': self.arena_size,
                'walls': self.arena.walls,
                'obstacles': self.arena.obstacles
            },
            'running': self.running,
            'paused': self.paused
        }
    
    def get_robot_data(self):
        robot_data = []
        for robot in self.robots:
            robot_data.append({
                'id': robot.robot_id,
                'state': robot.state,
                'position': robot.position,
                'orientation': robot.orientation,
                'zone': self.zones.index(robot.assigned_zone) if robot.assigned_zone else None,
                'coverage': robot.get_coverage(),
                'tof_hits': robot.tof_hits,
                'battery': robot.battery,
                'path': list(robot.path_history),
                'local_grid': robot.local_grid
            })
        return robot_data


def main():
    parser = argparse.ArgumentParser(description='Multi-Robot Swarm Mapping Simulation')
    parser.add_argument('--robots', type=int, default=3, help='Number of robots')
    parser.add_argument('--size', type=int, default=20, help='Arena size in meters')
    parser.add_argument('--obstacles', type=int, default=5, help='Number of obstacles')
    parser.add_argument('--headless', action='store_true', help='Run without UI')
    parser.add_argument('--steps', type=int, default=300, help='Maximum simulation steps')
    args = parser.parse_args()
    
    sim = Simulation(n_robots=args.robots, arena_size=args.size, n_obstacles=args.obstacles)
    sim.max_steps = args.steps
    
    if args.headless:
        print("\n=== Running Headless Simulation ===")
        while sim.running:
            completed = sim.update()
        print("\n=== Simulation Complete ===")
    else:
        try:
            from simulation_ui import SimulationUI
            ui = SimulationUI(sim)
            ui.run()
        except ImportError as e:
            print(f"UI not available ({e}), running headless...")
            while sim.running:
                completed = sim.update()
            print("\n=== Simulation Complete ===")


if __name__ == "__main__":
    main()