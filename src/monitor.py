import matplotlib.pyplot as plt

class Monitor:
    def __init__(self):
        self.rewards = [0]
        self.average_rewards = [0]
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.reward_line, = self.ax.plot(self.rewards, 'r-', label='Reward')
        self.avg_reward_line, = self.ax.plot(self.average_rewards, 'b-', label='25-Episode Moving Average')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward')
        self.ax.set_title('Real-time Reward Monitoring')
        self.ax.legend()

    def log_reward(self, reward):
        """Logs the received reward and updates moving average."""
        self.rewards.append(reward)

        # Update moving average
        if len(self.rewards) <= 25:
            average = sum(self.rewards) / len(self.rewards)
        else:
            average = sum(self.rewards[-25:]) / 25
        self.average_rewards.append(average)

    def update_plot(self):
        """Updates the plot with new data for rewards and moving average."""
        x_data = range(len(self.rewards))
        self.reward_line.set_data(x_data, self.rewards)
        self.avg_reward_line.set_data(x_data, self.average_rewards)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
