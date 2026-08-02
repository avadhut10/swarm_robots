import numpy as np

class KalmanFilter:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        self.Q = np.eye(4, dtype=np.float32) * 0.05
        self.R = np.eye(2, dtype=np.float32) * 0.3
        self.P = np.eye(4, dtype=np.float32) * 1.0
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.initialized = False

    def set_initial_state(self, position, velocity=None):
        x_pos, y_pos = position
        x_vel, y_vel = velocity if velocity is not None else (0.0, 0.0)
        self.x = np.array([[x_pos], [y_pos], [x_vel], [y_vel]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 0.1
        self.initialized = True
        print(f"🔍 [KalmanFilter] Initialized state to: ({x_pos:.2f}, {y_pos:.2f})")

    def predict(self):
        if not self.initialized: return np.array([0.0, 0.0])
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return np.array([self.x[0, 0], self.x[1, 0]], dtype=np.float32)

    def update(self, measurement):
        x_meas, y_meas = measurement
        z = np.array([[x_meas], [y_meas]], dtype=np.float32)
        if not self.initialized:
            self.set_initial_state((x_meas, y_meas))
            return np.array([x_meas, y_meas], dtype=np.float32)
            
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        S_inv = np.linalg.inv(S)
        K = np.dot(np.dot(self.P, self.H.T), S_inv)
        
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(4, dtype=np.float32) - np.dot(K, self.H), self.P)
        return np.array([self.x[0, 0], self.x[1, 0]], dtype=np.float32)
