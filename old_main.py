from src.display import Display
from src.drone import Drone
from src.human import Human
from src.agent import Agent
from src.monitor import Monitor
from src.utils import read_config
import time

def initialize():
    config = read_config("config.yaml")
    monitor = Monitor(config)
    monitor.update_plot()

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



    return display, monitor, drone, human, agent

def main():
    display, monitor, drone, human, agent = initialize()

    target = {
        "x": display.width // 2,
        "y": display.height // 2,
        "distance": 100
    }
    state, normalized_state = drone.get_state(), drone.get_normalized_state(target)
    
    total_reward = 0
    
    while True:
        previous_state = normalized_state
        #action = human.get_action()
        action = agent.get_action(state=drone.get_normalized_state(target))
        
        done = drone.update_state(inputs=action)
        state, normalized_state = drone.get_state(), drone.get_normalized_state(target)
        
        if agent.n_games % 10 == 0:
            display.update(drone, agent, target)
            #time.sleep(1)
        reward = agent.get_reward(state=state, target=target, done=done)
        total_reward += reward
        
        agent.train_short_memory(previous_state, action, reward, normalized_state, done)
        agent.remember(previous_state, action, reward, normalized_state, done)
        
        if done:
            monitor.log_data(total_reward, drone.survive_duration)
            monitor.update_plot()
            drone.reset_state()
            state, normalized_state = drone.get_state(), drone.get_normalized_state(target)
            agent.n_games += 1
            agent.train_long_memory()
            print(f"Game {agent.n_games} done, epsilon: {round(agent.epsilon*100,1)}, total reward: {round(total_reward)}")
            total_reward = 0
        

if __name__ == "__main__":
    main()