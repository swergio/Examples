# Example: Reinforcement Learning in Gym Environment

This example will show how to use the swergio project in an reinforcement learning setup. 

We will setup a RL agent that will perform actions in the *CartPole-v1* gym environment (https://gymnasium.farama.org/environments/classic_control/cart_pole/). 
The model component will communicate the predicted action to the environment component based on received observations and in return the environment will give a reward feedback.

In addition to the simple RL setup with only one agent and one environment, this example extends the setup to be able to select an action based on the input of multiple models. 
To perform this selection we define a separate component that will aggregate all model outputs to an combined action distribution.

In this example we will also try to further improve the model performance by performing an additional evolutionary algorithm to find the best RL agent. 
We therefore rank each of the models based on their contribution to the selected actions, then mate and mutate the best models to get a new set of agents.


## Requirements

To run this example we need to run python code, the easiest way to set it up is to:
   1. Install python > 3.8 (https://www.python.org/downloads/)
   2. If required install python3-venv ``sudo apt install python3-venv``
   3. Create a virtual environment in the example folder ``python3 -m venv /venv``
   4. Activate the virtual environment ``source \env\bin\activate``
   5. Install required packages to virtual environment ``pip install -r requirements.txt``


## How to run

To run this example including running the environment and performing a evolutionary step we can run separate code blocks either via interactive python in vscode from the ``06_control.py`` file or as jupyter notebook ``06_control.ipynb``.

The code will first start all necessary components and then define a client to communicate the commands to run episodes with the environment and to start a evolutionary step for the models.

Using this client we can write a loop to train the models via RL by running episodes and in between select better models via evolutionary algorithm. 

The final code block in the files will shut down all the components.