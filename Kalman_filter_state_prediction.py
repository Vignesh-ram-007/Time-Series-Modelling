import numpy as np
import matplotlib.pyplot as plt

# Define the KalmanFilter class
class KalmanFilter:
    def __init__(self, A, B, H, Q, R, x0, P0):
        self.A = A  # State transition matrix
        self.B = B  # Control input matrix (if applicable)
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance matrix
        self.R = R  # Measurement noise covariance matrix
        self.x = x0  # Initial state estimate
        self.P = P0  # Initial covariance estimate

    def predict(self, u=None):
        # Predict step
        self.x = np.dot(self.A, self.x)
        if u is not None:
            self.x += np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        # Update step
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x += np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)

# Example usage
if __name__ == '__main__':
    dt = 0.1  # Time step
    A = np.array([[1, dt],
                  [0, 1]])  # State transition matrix (constant velocity model)
    B = None  # No control input
    H = np.array([[1, 0]])  # Observation matrix (position observation)
    Q = np.array([[0.01, 0],
                  [0, 0.01]])  # Process noise covariance (tuned for the example)
    R = np.array([[0.1]])  # Measurement noise covariance (tuned for the example)
    x0 = np.array([[0],
                   [0]])  # Initial state estimate
    P0 = np.array([[1, 0],
                   [0, 1]])  # Initial covariance estimate

    kf = KalmanFilter(A=A, B=B, H=H, Q=Q, R=R, x0=x0, P0=P0)

    # Simulate measurements (position with noise)
    num_steps = 100
    true_positions = np.zeros((num_steps, 1))
    measurements = np.zeros((num_steps, 1))
    for t in range(num_steps):
        true_positions[t] = 2 * t * dt  # True position (linear motion)
        measurements[t] = true_positions[t] + np.random.normal(0, np.sqrt(R[0, 0]))  # Add measurement noise

    estimated_positions = np.zeros((num_steps, 1))
    for t in range(num_steps):
        kf.predict()
        kf.update(measurements[t])
        estimated_positions[t] = kf.x[0]

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(true_positions, label='True Position')
    plt.plot(measurements, label='Measured Position', marker='o', linestyle='None')
    plt.plot(estimated_positions, label='Estimated Position', linestyle='--')
    plt.title('Kalman Filter Example')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True)
    plt.show()
