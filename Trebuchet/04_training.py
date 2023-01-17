# %%[markdown]

# # Trainer

# This file contains the logic for the trainer of the network.
# The trainer provides random samples  of wind speed and target distance, the trebuchet should aim at.
# After running the information through the forward pass the trainer will evaluate the result of the trebuchet and provide a gradient of the difference between actual and achieved target as feedback.


# %%
import socket
import numpy as np
import uuid
import torch
from torch import nn
from swergio import Client, Trigger, MESSAGE_TYPE, MODEL_STATUS
from swergio_toolbox.objects import MutableNumber
# %%[markdown]

# We set the COMPONENT_NAME to trainer, which is use in the communication.

# We need to specify the IP and the PORT as well as the message FORMAT and the HEADER_LENGTH. 
# All of these information have to stay the same across the server and all clients.
# %%
COMPONENT_NAME = 'trainer'

PORT = 8080
SERVER = socket.gethostbyname(socket.gethostname())
FORMAT = 'utf-8'
HEADER_LENGTH = 10


# %%[markdown]

# For the trainer we first define a constant variable TRAIN_STEPS with the number of total training steps we want to run through once the script is executed.
# We then initiate a mutable number as step counter to track how many steps we've already been through. This variable needs to be mutable since we will update the value in side the handler function context.
# Including the step_counter we can initialize the swergio client.
# Finally we also define a loss function we will use to calculate the gradient between actual and achieved target als well as the memory dictionary we will use to store the send targets until we get the results.

# %%

TRAIN_STEPS = 100
step_counter = MutableNumber(0)

client = Client(COMPONENT_NAME,SERVER,PORT,FORMAT,HEADER_LENGTH,step_counter = step_counter)
l_fn = nn.MSELoss()
memory = {}
# %%[markdown]

# We define a function to generate a new message containing a random sampled wind speeds and target for the trebuchet.
# The random target is can be between 20 and 100. While the wind speed is sampled from a normal distribution with factor 5 and can therefore be negative or positive.
# After constructing the message as dictionary and adding the samples to the "DATA" entry, the function also stores the targe value in the component memory dict.
# This value is retrieved and used once we get the results from the trebuchet. 
# To identify the origin of the message throughout the whole process, we set a ROOT_ID which will be passed from message to message.

# %%
def new_msg():
    nr_of_samples = 100

    target = np.random.uniform(20,100,nr_of_samples)
    wind = np.random.randn(nr_of_samples)*5 #np.random.uniform(0,5,nr_of_samples)

    data_id = uuid.uuid4().hex
    d = {'ROOT_ID':data_id, 
        'ID':uuid.uuid4().hex,
        'TYPE': MESSAGE_TYPE.DATA.FORWARD.id, 
        'DATA': np.stack([wind,target],axis = 1).tolist(), 
        'TO_ROOM': 'input',
        'MODEL_STATUS': MODEL_STATUS.TRAIN.id
    } 
    memory[data_id] = target
    return d

# %%[markdown]

# Once the samples are processed by the trebuchet, the trainer needs to give feedback how accurate the results are.
# We therefore use a handler that is triggered by incoming messages in the 'output' room from type DATA.FORWARD.
# We extract the achieved target from the "DATA" entry in the message and retrieve the actual target from our internal memory based on the ROOT_ID of the message (same we provide in new_msg).
# After cleaning up the memory, we use both values to calculate a loss between the actual and achieved target.
# With the pytorch autograd functions we can also calculate the gradient for this loss in respect to the achieved target.
# This gradient information we will pass down to the trebuchet as our feedback by adding it to the message we send back to the 'output' room as DATA.GRADIENT type. 

# %%
def backward(msg):
    pred = torch.FloatTensor(msg["DATA"])
    if msg["ROOT_ID"] not in memory.keys():
        return
    y = memory[msg["ROOT_ID"]]
    y = torch.tensor(y).float()
    memory.pop(msg["ROOT_ID"])
    l = l_fn(pred,y)
    print(l)
    l_gradient = torch.autograd.functional.jacobian(lambda x: l_fn(x,y),pred)
    l_gradient  = l_gradient.tolist()
    return {"ID":msg["ID"],"DATA":l_gradient}

client.add_eventHandler(backward,MESSAGE_TYPE.DATA.GRADIENT,responseRooms='output',trigger=Trigger(MESSAGE_TYPE.DATA.FORWARD,'output'))
# %%[markdown]

# To automatically send a new batch of sample data to the network once the feedback of the prior batch is handled, we create a separate handler.
# The handler function itself just updates the step counter and checks if we did not exceed the total training steps. If that's the case it will send a new generated message including a new batch.
# The handler is triggered once it receives a DATA.GRADIENT message in the 'input' room, which will be the final gradient send by the control model.
# The "response" message is sent to the 'input' room, but as a new DATA.FORWARD message.

# %%
def new_data(msg, step_counter):
    step_counter += 1
    if step_counter.value < TRAIN_STEPS:
        d = new_msg()
        return  d
    
client.add_eventHandler(new_data,MESSAGE_TYPE.DATA.FORWARD,responseRooms='input',trigger=Trigger(MESSAGE_TYPE.DATA.GRADIENT,'input'))
# %%[markdown]

# At last the script will send the first message with a sampled batch to the 'input' room.
# Once we sent the message the client will start listening for responses from the other components.

# %%

client.send(new_msg())

client.listen()