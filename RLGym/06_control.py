# %%[markdown]

# # Control

# This script will start all other components as well as run the environment and perform a evolutionary step. 
# %%
import uuid
import socket
import time

from swergio  import Client, MESSAGE_TYPE
from swergio_toolbox.swarm_control import Swarm
# %%[markdown]

# We will use the swarm class from the swergio toolbox to simplify the handling of multiple components at once. 
# Of course we can also just start each component one by one by just running the according script.

# The swarm class requires a swarm.yaml file, that contains the specification about each component most importantly the path to each script.
# Once the YAML is defined we run each component at once by instantiating a swarm object and calling the start() method.

# %%
swarm = Swarm()
swarm.start()

# %%[markdown]

# To start all 20 models we'll just run a for loop and add each model to our swarm and start it. All the models refer to the same file, but are separate processes.

# %%
for i in range(20):
    swarm.add("model"+str(i), "02_model.py")
    swarm.start("model"+str(i))

# %%[markdown]

# After all the components are running, we can define another component that we will use a control unit by sending message in the 'control' room.
# We will set up a swergio client with a name and the same information as before and also join the 'control'

# %%
COMPONENT_NAME = 'control'

PORT = 8080
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
HEADER_LENGTH = 10

client = Client(COMPONENT_NAME,SERVER,PORT,FORMAT,HEADER_LENGTH)
client.join_room('control')

# %%[markdown]

# Now let's define a function we can use to run N number of episodes of the gym environment.
# This function will send the number of episodes as well as our deterministic flag to the 'control' room. Once the environment component will receive the message it will send observations.
# We can wait until we receive a feedback from the environment component that all episode are done including our final score the current models were able to reach.

# %%
def start(nr_of_episodes = 1, deterministic = False):
    msg = {'ROOT_ID':uuid.uuid4().hex, 
        'ID':uuid.uuid4().hex,
        'TYPE': MESSAGE_TYPE.DATA.CUSTOM.id, 
        'NR_OF_EPISODE': nr_of_episodes,
        'DETERMINISTIC': deterministic, 
        'TO_ROOM': 'control'
    } 
    client.send(msg)
    ## WAIT FOR RESPONSE
    while True:
        message = client.receive()
        if message is not False:
            if 'AVG_SCORE' in message.keys() and 'STATUS' in message.keys():
                if message['STATUS'] == 'ENV_DONE':
                    return  message['AVG_SCORE']
            
# %%[markdown]

# If we want to perform a evolutionary step, we will send a message including "EVOLUTION" True. The evolutionary component will then start to perform the evolution and update the model weights accordingly.
# Again we will wait until we receive feedback that this step is done.

# %%
def start_evo():
    msg = {'ROOT_ID':uuid.uuid4().hex, 
        'ID':uuid.uuid4().hex,
        'TYPE': MESSAGE_TYPE.DATA.CUSTOM.id, 
        'EVOLUTION': True,
        'TO_ROOM': 'control'
    } 
    client.send(msg)
    ## WAIT FOR RESPONSE
    while True:
        message = client.receive()
        if message is not False:
            if 'STATUS' in message.keys():
                if message['STATUS'] == 'EVO_DONE':
                    return  True
    
# %%[markdown]

# Let's run a few steps of episodes while training the models via RL and in between perform some evolutions to improve the model pool.
# In this example we will run 20 episodes none deterministic, meaning our actions are sampled and not necessary the best according to our policies.
# Then we can run 1 episode to check how well our models perform after RL training.
# Afterwards we start a evolutionary step and check again the performance of the now new model pool.

# %%
TRAIN_EVALS = 20
for i in range(5):
    print("ROUND " + str(i))
    score = start(TRAIN_EVALS,deterministic=False)
    print(score)
    score = start(1,deterministic=True)
    print(score)
    evdone = start_evo()
    print(evdone)
    time.sleep(1)
    score = start(1,deterministic=True)
    print(score)

# %%[markdown]

# Finally we can stop all the components (Server,ControlModel,Trebuchet and Trainer) all  at once with  the swarm.stop() method.
# %%
swarm.stop()


# %%
