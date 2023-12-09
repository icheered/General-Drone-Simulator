#
# The LoggerCallback class is used to log the reward and survive time of each episode and update the plot.
# It is called automatically by the training loop on every step
#

from stable_baselines3.common.callbacks import BaseCallback
from src.monitor import Monitor

class LoggerCallback(BaseCallback):
    def __init__(self, monitor: Monitor, verbose=0):
        super(LoggerCallback, self).__init__(verbose)
        self.monitor = monitor
        self.episode_reward = 0

    def _on_step(self) -> bool:
        # Accumulate rewards
        self.episode_reward += self.locals["rewards"][0]

        # Check if the episode is done
        if self.locals["dones"][0]:
            survive_duration = self.locals["infos"][0].get("episode_step", 0)
            self.monitor.log_data(self.episode_reward, survive_duration)
            self.monitor.update_plot()

            # Reset the reward for the next episode
            self.episode_reward = 0

        return True