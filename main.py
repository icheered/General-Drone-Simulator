from src.display import Display
from src.drone import Drone
from src.human import Human
from src.utils import read_config


def main():
    config = read_config("config.yaml")
    display = Display(
        config=config["display"],
        update_frequency=config["display"]["update_frequency"],
        title="Drone Simulation"
    )
    drone = Drone(
        config=config["drone"],
        update_frequency=config["display"]["update_frequency"],
        startx=config["display"]["width"] // 2,
        starty=config["display"]["height"] // 2,
    )

    human = Human(
        input_length=len(config["drone"]["motors"]),
    )

    # Do: Create agent
    while True:
        # Get input
        action = human.get_action()
        drone.update_state(inputs=action)
        drone.print_state()
        display.update(drone)

    
    pass

if __name__ == "__main__":
    main()