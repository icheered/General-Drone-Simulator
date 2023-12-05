from src.display import Display
from src.drone import Drone
from src.human import Human
from src.agent import Agent
from src.monitor import Monitor
from src.utils import read_config

def initialize():
    config = read_config("config.yaml")
    display = Display(
        config=config["display"],
        update_frequency=config["display"]["update_frequency"],
        title="Drone Simulation"
    )
    drone = Drone(
        config=config["drone"],
        display=config["display"],
        update_frequency=config["display"]["update_frequency"],
        startx=config["display"]["width"] // 2,
        starty=config["display"]["height"] // 2,
    )

    human = Human(
        input_length=len(config["drone"]["motors"]),
    )

    agent = Agent(
        drone=drone,
        config=config
    )

    monitor = Monitor()

    return display, monitor, drone, human, agent

def main():
    display, monitor, drone, human, agent = initialize()

    state, normalized_state = drone.get_state(), drone.get_normalized_state()
    
    total_reward = 0
    target = {
        "x": display.width // 2,
        "y": display.height // 2,
        "distance": 100
    }
    while True:
        previous_state = normalized_state
        #action = human.get_action()
        action = agent.get_action(state=drone.get_normalized_state())
        
        state, normalized_state, done = drone.update_state(inputs=action)
        
        if agent.n_games % 10 == 0:
            display.update(drone, target)
        reward = agent.get_reward(state=state, target=target, done=done)
        total_reward += reward
        
        agent.train_short_memory(previous_state, action, reward, normalized_state, done)
        agent.remember(previous_state, action, reward, normalized_state, done)
        
        if done:
            state, normalized_state = drone.reset_state()
            agent.n_games += 1
            agent.train_long_memory()
            print(f"Game {agent.n_games} done, epsilon: {round(agent.epsilon*100,1)}, total reward: {round(total_reward)}")
            monitor.log_reward(total_reward)
            monitor.update_plot()
            total_reward = 0
        

if __name__ == "__main__":
    main()