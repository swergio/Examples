# %%[markdown]

# # Run and Inference

# This script will start all other components, start the training and enables us afterwards to run some inference on the trained model.
# %%
import multiprocessing
import socket
import numpy as np
import uuid

from swergio import Client, Trigger, MESSAGE_TYPE, MODEL_STATUS
from swergio_toolbox.swarm_control import Swarm
# %%[markdown]

# We will use the swarm class from the swergio toolbox to simplify the handling of multiple components at once. 
# Of course we can also just start each component one by one by just running the according script.

# The swarm class requires a swarm.yaml file, that contains the specification about each component most importantly the path to each script.
# Once the YAML is defined we run each component at once by instantiating a swarm object and calling the swarm.start() method.

# %%
swarm = Swarm()
swarm.start()
# %%[markdown]

# In this exampled we did not add the trainer to the YAML since we want to b flexible, when to start the component independently.
# We can now easily add the trainer to the swarm by providing a name and the path to the source file.

# %%
swarm.add("trainer", "04_training.py", terminal_cmd="")

# %%[markdown]

# Once added to the swarm we can start a specific component by name. In our case we now start the trainer.
# The trainer will now send the first sample message to the network and will continue until the end of the set training steps.
# %%
swarm.start("trainer")

# %%[markdown]

# Since each of the components is independent we can already send additional inference message to the network or we can wait until the trainer is done.
# For inference we can use a similar set-up as for the trainer. We will generate some data including wind speed and desired target and send them to  the 'input' room.
# The other components will pick up the message and will provide us with an answer in the 'output' room.

# To be able to send message we will set up a swergio client with a name and the same information as before.

# %%
COMPONENT_NAME = 'inference'

PORT = 8080
SERVER = socket.gethostbyname(socket.gethostname())
FORMAT = 'utf-8'
HEADER_LENGTH = 10

client = Client(COMPONENT_NAME,SERVER,PORT,FORMAT,HEADER_LENGTH)

# %%[markdown]

# Similar to the trainer we define a function that generates us a new message including the sampled data for wind speed and target.

# %%

def new_msg():
    nr_of_samples = 100
    target = np.random.uniform(20,100,nr_of_samples)
    wind =  np.random.randn(nr_of_samples)*5

    data_id = uuid.uuid4().hex
    d = {'ROOT_ID':data_id, 
        'ID':uuid.uuid4().hex,
        'TYPE': MESSAGE_TYPE.DATA.FORWARD.id, 
        'DATA': np.stack([wind,target],axis = 1).tolist(), 
        'TO_ROOM': 'input',
        'MODEL_STATUS': MODEL_STATUS.TRAIN.id
    } 
    return d
# %%[markdown]

# Once we receive a result in the 'output' room we want to print the message. 
# We therefore add an event handler to our client which is trigger by a DATA.FORWARD message in 'output' and prints the message.
# We still need to define a response TYPE and ROOM for the handler, but our function will always return None and therefore never send a message itself.
# %%
def print_response(msg):
    print(msg)
    return None 

client.add_eventHandler(print_response,MESSAGE_TYPE.DATA.FORWARD,responseRooms='output',trigger=Trigger(MESSAGE_TYPE.DATA.FORWARD,'output'))

# %%[markdown]

# In case we run this script as notebook to be able to send multiple inference message, we don't want to block our process while listening for the result.
# We can therefore start the client.listen() method as multiprocess and are able to execute further notebook cells.
# %%
multiprocessing.Process(target=client.listen).start()
# %%[markdown]

# Now we can create a new message and send in with our client to the other components.  
# Once we receive the result the handler will print it in the listen process.
# %%
req = new_msg()
print(req)
client.send(req)
# %%[markdown]

# Finally we can stop all the running components (Server,ControlModel,Trebuchet and Trainer) at once with  the swarm.stop() method.
# %%
swarm.stop()
# %%
