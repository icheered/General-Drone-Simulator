# This file is not used anymore but shows a possible (not yet working implementation) for a custom callback
# that receives the trajectory of the drone each time env.step() is called
#
#
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class LogTrajectoriesCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.trajectories = []

    def _on_step(self) -> bool:
        # Get the current state and action from the environment
        # obs = self.model.env.get_attr("get_obs")()
        # actions = self.model.env.get_attr("get_actions")()
        # mass = self.model.env.mass  # Access mass directly from the environment

        # # Store the state-action pairs in a trajectory
        # trajectory = {"states": np.copy(obs), "actions": np.copy(actions), "mass": np.copy(mass)}
        # self.trajectories.append(trajectory)
        self.trajectories.append({"1": 1})

        return True