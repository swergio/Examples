# Example: Melody Creator

In this YouTube video (https://www.youtube.com/watch?v=aOsET8KapQQ) *Kie Codes* showed an example on how to generate a melody with an genetic algorithm. The full code can be found on github https://github.com/kiecodes/generate-music.

In Kies example you have to evaluate all generated melodies by a user, to give feedback to the genetic algorithm. 
With this example we want to extend Kies approach, by using a separate evaluation network that can predict a rating for each melody and therefore enables the evolutionary algorithm to perform faster.

In addition we will live train the evaluation network with an active learning approach. 
Which means for melodies the network is uncertain of to predict a rating, we will ask the user to provide us with labels and then re-train the network with this  newly labeled data.

Since our new setup requires several independent component (Evolutionary Algorithm, Evaluation Network and Active Learner) we will see how we can leverage swergio to enable all necessary communication.


## Requirements

For this example we need to run python code and the easiest way to set it up is to:
   1. Install python > 3.8 (https://www.python.org/downloads/)
   2. If required install python3-venv ``sudo apt install python3-venv``
   3. Create a virtual environment in the example folder ``python3 -m venv /venv``
   4. Activate the virtual environment ``source \env\bin\activate``
   5. Install required packages to virtual environment ``pip install -r requirements.txt``


## How to run

To run this example including training our evaluation network and provide active labels for new data, we can run separate code blocks either via interactive python in vscode from the ``05_user_interface.py`` file or as jupyter notebook ``05_user_interface.ipynb``.

The script will start all required components and the setup another swergio client to communicate the labeling of the new data as well as sending the starting command.

The final code block in the files will shut down all the components.