import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Import our custom swarm simulation modules
from kalman_filter import KalmanFilter
from collision_avoidance import CollisionAvoidance
from aruco_detector import ArUcoDetector
from robot_manager import RobotManager
from simulation_ui import SimulationUI

class SwarmSimulatorApp:
    def __init__(self):
        # 1. Initialize core system modules
        # Camera ID 1 (Motorola Smart Connect). Fallback is handled inside detector.
        self.detector = ArUcoDetector(camera_id=1)
        self.robot_manager = RobotManager(scale_factor=0.01)
        self.ui = SimulationUI()
        
        # Connect UI with manager
        self.ui.set_manager(self.robot_manager)
        
        # 2. Kalman filter instance per robot ID (100, 101, 102)
        self.kalman_filters = {
            100: KalmanFilter(dt=0.1),
            101: KalmanFilter(dt=0.1),
            102: KalmanFilter(dt=0.1)
        }
        
        # 3. Connect UI Button and canvas interactions
        self.ui.btn_calc.on_clicked(self.calculate_paths)
        self.ui.btn_anim.on_clicked(self.toggle_animation)
        self.ui.btn_reset.on_clicked(self.reset_all)
        self.ui.btn_clear.on_clicked(self.clear_jobs)
        self.ui.register_click_callback(self.add_manual_job)
        
        # Simulation animation states
        self.animating = False
        self.ani = None
        self.dt = 0.1
        self.robot_speed = 0.4
        
        # For adding simulated jobs manually when clicking on canvas
        self.next_job_marker_id = 22

    def add_manual_job(self, x, y):
        """Allows manual addition of a simulated job at clicked coordinates."""
        marker_id = self.next_job_marker_id
        self.next_job_marker_id += 1
        
        # Create a mock detection snapshot to append this job
        positions = self.detector.get_positions()
        # Scale back up to mm
        x_mm = x / self.robot_manager.scale_factor
        y_mm = y / self.robot_manager.scale_factor
        
        positions['tasks'][marker_id] = (x_mm, y_mm)
        
        # Update robot manager with modified dictionary
        self.robot_manager.update_from_detection(positions)
        self.ui.draw()
        self.ui.update_status()
        print(f"📦 [Main] Added manual Job J{marker_id-19} at simulation coords: ({x:.2f}, {y:.2f})")

    def calculate_paths(self, event):
        """
        Path planning: Assigns jobs to the nearest robots and routes them.
        Each robot's waypoints path will be:
        1. Robot Current Pos -> 2. Assigned Job Pos (Pick) -> 3. END Marker Pos (Deliver)
        """
        print("\n🔄 [PathPlanner] Starting path calculations and jobs assignment...")
        
        robots = self.robot_manager.get_robots()
        jobs = [j for j in self.robot_manager.get_jobs() if not j['picked']]
        end_pos = self.robot_manager.get_end()
        
        if len(robots) == 0:
            print("⚠️ [PathPlanner] No active robots detected. Cannot calculate paths!")
            return
            
        if len(jobs) == 0:
            print("⚠️ [PathPlanner] No pending jobs detected. Cannot calculate paths!")
            return
            
        if end_pos is None:
            print("⚠️ [PathPlanner] END marker is missing. Please place the END marker (ID 11) to plan delivery paths.")
            return

        # Clear existing waypoints
        for robot in robots:
            robot['waypoints'] = []
            robot['assigned_jobs'] = []
            robot['path'] = [list(robot['pos'])]

        # Greedy nearest-neighbor job assignment
        available_jobs = list(jobs)
        for job in available_jobs:
            # Find the nearest robot to this job
            best_robot = None
            min_dist = float('inf')
            
            for r in robots:
                # We prioritize robots with fewer assigned jobs to balance the load
                load_penalty = len(r['assigned_jobs']) * 10.0
                dist = np.linalg.norm(np.array(r['pos']) - np.array(job['pos'])) + load_penalty
                if dist < min_dist:
                    min_dist = dist
                    best_robot = r
                    
            if best_robot is not None:
                job['assigned_to'] = best_robot['id']
                best_robot['assigned_jobs'].append(job)
                print(f"🎯 [PathPlanner] Assigned Job {job['id']} to Robot {best_robot['id']} (Distance: {min_dist:.2f} units)")

        # Create sequential waypoint paths for each robot
        for r in robots:
            if len(r['assigned_jobs']) > 0:
                # Target waypoints: Pick up each job sequentially, then deliver to END
                wps = []
                for job in r['assigned_jobs']:
                    wps.append(job['pos'])
                wps.append(end_pos)
                
                r['waypoints'] = wps
                print(f"🗺️ [PathPlanner] Robot {r['id']} waypoints path planned: {len(wps)} steps to final delivery.")
            else:
                # If no jobs assigned, stay put or go to START / END
                print(f"💤 [PathPlanner] Robot {r['id']} has no jobs assigned.")

        self.ui.draw()
        self.ui.update_status()

    def toggle_animation(self, event):
        """Starts/Pauses active simulation animation."""
        self.animating = not self.animating
        self.ui.is_animating = self.animating
        
        if self.animating:
            self.ui.btn_anim.label.set_text("⏸️ Pause Animation")
            self.ui.btn_anim.color = '#F1C40F'
            self.ui.btn_anim.hovercolor = '#F39C12'
            print("▶️ [Main] Animation Started.")
        else:
            self.ui.btn_anim.label.set_text("▶️ Start Animation")
            self.ui.btn_anim.color = '#2ECC71'
            self.ui.btn_anim.hovercolor = '#27AE60'
            print("⏸️ [Main] Animation Paused.")
            
        self.ui.update_status()

    def reset_all(self, event):
        """Resets all simulation assignments and paths."""
        print("🔄 [Main] Resetting simulation paths and assignments...")
        self.animating = False
        self.ui.is_animating = False
        self.ui.btn_anim.label.set_text("▶️ Start Animation")
        self.ui.btn_anim.color = '#2ECC71'
        
        # Re-fetch raw camera snapshot to reset clean positions
        positions = self.detector.get_positions()
        self.robot_manager.update_from_detection(positions)
        
        self.ui.draw()
        self.ui.update_status()

    def clear_jobs(self, event):
        """Wipes out all jobs and calculations."""
        print("🗑️ [Main] Clearing all active jobs...")
        self.animating = False
        self.ui.is_animating = False
        self.ui.btn_anim.label.set_text("▶️ Start Animation")
        self.ui.btn_anim.color = '#2ECC71'
        
        # Clear positions dictionary in detector & update manager
        with self.detector.lock:
            self.detector.task_positions = {}
            
        positions = self.detector.get_positions()
        self.robot_manager.update_from_detection(positions)
        
        self.ui.draw()
        self.ui.update_status()

    def update(self, frame):
        """
        Main tick handler.
        ALWAYS fetches from detector (even when paused) to keep positions updated.
        """
        # 1. Fetch current positions from ArUco detector snapshot
        positions = self.detector.get_positions()
        
        # Apply Kalman Filter to smooth detected robot coordinates
        for marker_id, pos_mm in list(positions['robots'].items()):
            if marker_id in self.kalman_filters:
                kf = self.kalman_filters[marker_id]
                filtered = kf.update(pos_mm)
                positions['robots'][marker_id] = (float(filtered[0]), float(filtered[1]))

        # Update manager positions (only detects and draws what is currently on screen)
        self.robot_manager.update_from_detection(positions)
        
        # 2. If animation is running, advance robots towards active waypoints
        if self.animating:
            robots = self.robot_manager.get_robots()
            all_robot_positions = [r['pos'] for r in robots]
            
            # Establish static obstacles list (START, jobs, boundary elements)
            obstacles = []
            start_pos = self.robot_manager.get_start()
            if start_pos:
                obstacles.append(start_pos)
            for j in self.robot_manager.get_jobs():
                if not j['picked']:
                    obstacles.append(j['pos'])

            for r in robots:
                if len(r['waypoints']) > 0:
                    current_pos = np.array(r['pos'], dtype=np.float32)
                    target = np.array(r['waypoints'][0], dtype=np.float32)
                    
                    # Compute standard heading direction vector
                    direction = target - current_pos
                    distance = np.linalg.norm(direction)
                    
                    if distance < 0.25:
                        # Reached current waypoint
                        reached_wp = r['waypoints'].pop(0)
                        print(f"🏁 [Main] Robot {r['id']} reached waypoint: ({reached_wp[0]:.2f}, {reached_wp[1]:.2f})")
                        
                        # Check if this waypoint matches any assigned job (Mark as picked)
                        for job in r['assigned_jobs']:
                            if np.linalg.norm(np.array(job['pos']) - np.array(reached_wp)) < 0.3:
                                job['picked'] = True
                                print(f"📦 [Main] Robot {r['id']} PICKED UP Job {job['id']}!")
                        continue
                        
                    # Standard navigation velocity vector
                    vel_dir = direction / (distance + 1e-5)
                    nominal_vel = vel_dir * self.robot_speed
                    
                    # Get positions of other robots for collision avoidance
                    other_positions = [p for p in all_robot_positions if not np.array_equal(p, r['pos'])]
                    
                    # Compute avoidance steering forces
                    avoid_force = CollisionAvoidance.compute_avoidance_force(
                        current_pos, nominal_vel, other_positions, obstacles,
                        safety_radius=2.0, max_force=12.0
                    )
                    
                    # Total command velocity combining steering and avoidance force
                    command_vel = nominal_vel + avoid_force
                    
                    # Update simulated coordinates
                    new_pos = current_pos + command_vel * self.dt
                    r['pos'] = list(new_pos)
                    
                    # Append coordinates to trace path history lines
                    r['path'].append(list(new_pos))

        # 3. Draw updated frames
        self.ui.draw()
        self.ui.update_status()
        return []

    def run(self):
        """Bootstraps continuous detection thread and Matplotlib loop."""
        # Start ArUco detector threaded camera captures
        self.detector.start_detection()
        
        
        # Start the Matplotlib FuncAnimation loop (10 FPS for lightweight, smooth UI updates)
        self.ani = FuncAnimation(self.ui.fig, self.update, interval=100, cache_frame_data=False)
        plt.show()
        
        # Clean shutdown after matplotlib window closes
        self.detector.stop_detection()

if __name__ == "__main__":
    app = SwarmSimulatorApp()
    app.run()