from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class StopTrainingOnMovingAverageReward(BaseCallback):
    def __init__(self, reward_threshold: float, window_size: int = 25, verbose: int = 0):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.window_size = window_size
        self.rewards = []

    def _on_step(self) -> bool:
        # Retrieve the latest evaluation reward
        latest_reward = self.locals.get('rewards', None)
        
        # Update the rewards list
        if latest_reward is not None:
            self.rewards.append(latest_reward)
            if len(self.rewards) > self.window_size:
                self.rewards.pop(0)

        # Compute the moving average
        if len(self.rewards) == self.window_size:
            moving_average = np.mean(self.rewards)

            # Check if moving average exceeds the threshold
            if moving_average >= self.reward_threshold:
                if self.verbose > 0:
                    print(f"Stopping training as moving average reward {moving_average:.2f} is above the threshold {self.reward_threshold:.2f}")
                return False

        return True
