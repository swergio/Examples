# Example: Trebuchet

This is an example on how we can use the swergio project to control the accuracy of a medieval trebuchet. 
It is inspired by the example provided by the zygote.jl team (https://github.com/JuliaComputing/ODSC2019/blob/master/03-Differentiable-trebuchet-Zygote.ipynb), which is completely in julia.

For our example we will only have the trebuchet logic itself as differentiable julia code, while all other components especially the control network will be in python/pytorch.

In our example we will train a control network to provide the best angel and counterweight for the trebuchet to aim at a given target with a given wind speed.

We can then run inference on the trained model to aim at never seen targets.

## Requirements

To run this example we need to run python as well as julia code. 

For the python part of the code the easiest way to set it up is to:
   1. Install python > 3.8 (https://www.python.org/downloads/)
   2. If required install python3-venv ``sudo apt install python3-venv``
   3. Create a virtual environment in the example folder ``python3 -m venv /venv``
   4. Activate the virtual environment ``source \env\bin\activate``
   5. Install required packages to virtual environment ``pip install -r requirements.txt``

To run the julia part of the example we will have to install julia as well as instantiate the project environment:
   1. Install Julia (https://julialang.org/downloads/) 
   2. Activate the project environment in the example folder ``(v1.0) pkg> activate .``
   3. Instantiate the project environment ``(Trebuchet) pkg> instantiate``

## How to run

To run this example including training the control network as well as being able to send inference data, we can run separate code blocks either via interactive python in vscode from the ``05_run_and_inference.py`` file or as jupyter notebook ``05_run_and_inference.ipynb``.

The code is split into one part starting all components as well as starting the training and a second part where we will set up a new client to communicate the inference data.
Since the components are independent we can run training and testing in parallel, but it might be advisable to wait to run the inference code until the training is finished to get more reasonable results.

The final code block in the files will shut down all the components.