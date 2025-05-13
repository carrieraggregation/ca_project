import os
import numpy as np

class SimulatorEnv:
    """
    Offline environment wrapper using precomputed dataset (X.npy, y.npy).
    Provides a minimal OpenAI Gym-like interface: reset(), step(action).
    """
    def __init__(self, data_dir=None):
        # Determine data directory (default two levels up into python/data)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, 'data'))
        data_dir = data_dir or default_dir

        # Load features and labels
        self.X = np.load(os.path.join(data_dir, 'X.npy'))  # shape [N, feat_dim]
        self.y = np.load(os.path.join(data_dir, 'y.npy'))  # shape [N, numCC]

        self.num_samples = self.X.shape[0]
        self.state_size  = self.X.shape[1]
        self.numCC       = self.y.shape[1]
        self.action_size = 2 ** self.numCC

        self.current = 0

    def reset(self):
        """Reset environment to start of dataset."""
        self.current = 0
        return self.X[self.current]

    def step(self, action):
        """
        Take an action (mask index), return next state, reward, done, info.
        Reward is negative sum absolute error between masked CQI prediction and true CQI.
        """
        # Clip action to valid range
        idx = action % self.action_size
        # Decode mask from action
        mask = np.array(list(np.binary_repr(idx, self.numCC)), dtype=int)
        # Predicted CQI = mask * true CQI (simple proxy)
        true_cqi = self.y[self.current]
        pred_cqi = mask * true_cqi
        # Reward: negative absolute error
        reward = -np.sum(np.abs(true_cqi - pred_cqi))

        # Advance
        self.current += 1
        done = (self.current >= self.num_samples)
        next_state = self.X[self.current] if not done else None
        info = {}
        return next_state, reward, done, info

    def action_space_sample(self):
        """Random action sampling."""
        return np.random.randint(self.action_size)

# Example usage
if __name__ == '__main__':
    env = SimulatorEnv()
    s0 = env.reset()
    print('Initial state shape:', s0.shape)
    for _ in range(5):
        a = env.action_space_sample()
        ns, r, d, _ = env.step(a)
        print(f'action={a}, reward={r:.2f}, next_state_shape={(ns.shape if ns is not None else None)}, done={d}')
