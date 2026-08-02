class RobotManager:
    def __init__(self, scale_factor=0.01):
        self.scale_factor = scale_factor
        self.robots = {}        # marker_id -> Robot Dict
        self.jobs = {}          # marker_id -> Job Dict
        self.start_marker = None
        self.end_marker = None
        
        self.robot_colors = {
            100: '#3498DB',  # Blue (R1)
            101: '#E74C3C',  # Red (R2)
            102: '#2ECC71'   # Green (R3)
        }
        self.job_colors = {
            20: '#E67E22',   # Orange (J1)
            21: '#9B59B6'    # Magenta/Purple (J2)
        }

    def update_from_detection(self, positions):
        """
        Updates robots, jobs, start, and end marker positions from real-time detection.
        CRITICAL: Wipes out any inactive or non-detected components to maintain absolute realism.
        """
        prev_robots = self.robots
        prev_jobs = self.jobs
        
        # 1. CLEAR ALL active robots, jobs, and markers
        self.robots = {}
        self.jobs = {}
        self.start_marker = None
        self.end_marker = None
        
        # 2. Process robots (IDs 100, 101, 102)
        detected_robots = positions.get('robots', {})
        for marker_id, pos_mm in detected_robots.items():
            sim_x = pos_mm[0] * self.scale_factor
            sim_y = pos_mm[1] * self.scale_factor
            r_id = f"R{marker_id - 99}"
            color = self.robot_colors.get(marker_id, '#7F8C8D')
            
            path = []
            waypoints = []
            assigned_jobs = []
            if marker_id in prev_robots:
                path = prev_robots[marker_id].get('path', [])
                waypoints = prev_robots[marker_id].get('waypoints', [])
                assigned_jobs = prev_robots[marker_id].get('assigned_jobs', [])
                
            self.robots[marker_id] = {
                'id': r_id,
                'marker_id': marker_id,
                'start': (sim_x, sim_y),
                'end': (sim_x + 0.5, sim_y + 0.5),
                'pos': [sim_x, sim_y],
                'color': color,
                'path': path,
                'waypoints': waypoints,
                'assigned_jobs': assigned_jobs,
                'real_pos_mm': pos_mm
            }
            
            if marker_id not in prev_robots:
                print(f"✅ [RobotManager] Robot {r_id} (ID {marker_id}) ADDED at ({sim_x:.2f}, {sim_y:.2f})")
            else:
                prev_pos = prev_robots[marker_id]['pos']
                if abs(prev_pos[0] - sim_x) > 0.1 or abs(prev_pos[1] - sim_y) > 0.1:
                    print(f"📍 [RobotManager] Robot {r_id} moved to ({sim_x:.2f}, {sim_y:.2f})")

        for marker_id in prev_robots:
            if marker_id not in self.robots:
                print(f"❌ [RobotManager] Robot R{marker_id - 99} REMOVED (not detected)")

        # 3. Process jobs (IDs 20, 21)
        detected_jobs = positions.get('tasks', {})
        for marker_id, pos_mm in detected_jobs.items():
            sim_x = pos_mm[0] * self.scale_factor
            sim_y = pos_mm[1] * self.scale_factor
            j_id = f"J{marker_id - 19}"
            color = self.job_colors.get(marker_id, '#F1C40F')
            
            picked = False
            assigned_to = None
            if marker_id in prev_jobs:
                picked = prev_jobs[marker_id].get('picked', False)
                assigned_to = prev_jobs[marker_id].get('assigned_to', None)
                
            self.jobs[marker_id] = {
                'id': j_id,
                'marker_id': marker_id,
                'pos': (sim_x, sim_y),
                'pos_mm': pos_mm,
                'color': color,
                'picked': picked,
                'assigned_to': assigned_to
            }
            if marker_id not in prev_jobs:
                print(f"✅ [RobotManager] Job {j_id} (ID {marker_id}) ADDED at ({sim_x:.2f}, {sim_y:.2f})")

        for marker_id in prev_jobs:
            if marker_id not in self.jobs:
                print(f"❌ [RobotManager] Job J{marker_id - 19} REMOVED (not detected)")

        # 4. Process START marker (ID 10)
        start_pos_mm = positions.get('start', None)
        if start_pos_mm is not None:
            self.start_marker = (start_pos_mm[0] * self.scale_factor, start_pos_mm[1] * self.scale_factor)
        else:
            self.start_marker = None

        # 5. Process END marker (ID 11)
        end_pos_mm = positions.get('end', None)
        if end_pos_mm is not None:
            self.end_marker = (end_pos_mm[0] * self.scale_factor, end_pos_mm[1] * self.scale_factor)
        else:
            self.end_marker = None

    def get_robots(self):
        return list(self.robots.values())

    def get_jobs(self):
        return list(self.jobs.values())

    def get_start(self):
        return self.start_marker

    def get_end(self):
        return self.end_marker
