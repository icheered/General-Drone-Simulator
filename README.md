# General Drone Simulator
## Installation
Hey from Rafat.
This project uses !(Poetry)[https://python-poetry.org/] for dependency management. Install poetry and run `poetry install` to install all dependencies. 
If you don't want to use poetry you can install the dependencies manually using pip.

### Configuration
Configuration of the drone environment can be done in the `config.yaml` file. Here you can change the motors of drones, the size of the environment, gravity, etc.
Further configuration is done in the respective python files discussed below.

### Manual control
Run `poetry run python manual.py` to manually control the drone. This is nice for tuning parameters in `config.yaml` and for testing the drone.

### Training
Training can be done through the Jupyter notebook `main.ipynb`. However, for some reason the graphs are broken so its hard to monitor training.
You can also run `poetry run python main.py` to train the model. Make sure to inspect the file and change the parameters to your liking.
During training, the best model is continuously saved and updated as `training/saved_models/best_model.zip`. When finishing or stopping training, the final model is also saved in the same folder.

The file `run_best_model.py` can be used *while* training (in a seperate terminal) to run the best model so far. This doesn't improve training, but it is really fun to watch the drone fly around and improve over time.

## Installing pytorch for GPU (NOT ADVISED)
Reinforcement learning does not benefit from GPU acceleration as much as other machine learning tasks. However, if you want to use GPU acceleration, you can install pytorch for rocm (AMD) using the following commands:

AMD (RocM)
```bash
poetry source add --priority=explicit pytorch-gpu-src https://download.pytorch.org/whl/rocm5.6
poetry add --source pytorch-gpu-src torch
```

Nvidia (CUDA)
```bash
poetry source add --priority=explicit pytorch-gpu-src https://download.pytorch.org/whl/cu118
poetry add --source pytorch-gpu-src torch
```

Note: You might have to mess around with the `pyproject.toml` and deleting the `.venv` folder to get this to work. I don't recommend it, but it is possible.