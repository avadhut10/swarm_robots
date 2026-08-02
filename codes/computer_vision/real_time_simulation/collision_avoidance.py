import numpy as np
from scipy.spatial.distance import euclidean

class CollisionAvoidance:
    @staticmethod
    def compute_avoidance_force(robot_pos, robot_vel, other_robots, obstacles,
                                safety_radius=2.0, max_force=15.0):
        robot_pos = np.array(robot_pos, dtype=np.float32)
        avoidance_force = np.zeros(2, dtype=np.float32)
        
        for other_pos in other_robots:
            other_pos = np.array(other_pos, dtype=np.float32)
            dist = euclidean(robot_pos, other_pos)
            if dist < 0.01:
                avoidance_force += np.random.uniform(-1.0, 1.0, 2) * max_force * 0.5
                continue
            if dist < safety_radius:
                force_mag = ((safety_radius - dist) / (dist + 1e-5)) * 5.0
                direction = (robot_pos - other_pos) / dist
                avoidance_force += direction * force_mag

        for obs_pos in obstacles:
            obs_pos = np.array(obs_pos, dtype=np.float32)
            dist = euclidean(robot_pos, obs_pos)
            if dist < 0.01:
                avoidance_force += np.random.uniform(-1.0, 1.0, 2) * max_force * 0.5
                continue
            if dist < safety_radius:
                force_mag = ((safety_radius - dist) / (dist + 1e-5)) * 6.0
                direction = (robot_pos - obs_pos) / dist
                avoidance_force += direction * force_mag

        force_magnitude = np.linalg.norm(avoidance_force)
        if force_magnitude > max_force:
            avoidance_force = (avoidance_force / force_magnitude) * max_force
            
        return avoidance_force
