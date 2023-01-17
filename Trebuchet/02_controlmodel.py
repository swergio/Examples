# %%[markdown]

# # Control Model

# This file contains the logic for the control neural network, 
# that will provide the angle and contra weight for the trebuchet based on given wind speed and target distance.

# %%
import torch 
from torch import nn
import uuid
import socket
from swergio  import Client, Trigger, MESSAGE_TYPE

# %%[markdown]

# We set the COMPONENT_NAME to controlmodel, which is use in the communication.

# We need to specify the IP and the PORT as well as the message FORMAT and the HEADER_LENGTH. 
# All of these information have to stay the same across the server and all clients.

# %%
COMPONENT_NAME = 'controlmodel'

PORT = 8080
SERVER = socket.gethostbyname(socket.gethostname())
FORMAT = 'utf-8'
HEADER_LENGTH = 10

# %%[markdown]

# We define the neural network structure. The network will take the wind speed and the target distance as concatenated input.
# It will output 3 dimension. The first is the wind speed, which we will forward to the next component. 
# The second is the logits for the angle of the trebuchet and last as third dimension the logits of the counter weight. 

# %%
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.Sigmoid(),
            nn.Linear(16, 64),
            nn.Sigmoid(),
            nn.Linear(64, 16),
            nn.Sigmoid(),
            nn.Linear(16, 2)
        )

    # input wind, target 
    def forward(self, x):
        logits = self.layers(x)
        # add wind to output
        output = torch.concat([x[:,0:1], logits], dim = -1)
        return output

# %%[markdown]

# Now we can instantiate the required objects.
# First the prior defined model as well as the optimizer to train the model.
# Additional we define a empty dictionary that will store the forward messages as memory. We will use the stored information to calculate the backward pass of our model once we receive the gradient feedback.
# Lastly we create the swergio Client by passing the required settings (NAME, SERVER, PORT etc. ) as well as the prior defined objects as keyword arguments, so they can be refereed to in our handling functions.

# %%

model = NeuralNetwork()
optimizer = torch.optim.Adam(model.parameters())
memory = {}
client = Client(COMPONENT_NAME, SERVER,PORT,FORMAT,HEADER_LENGTH,model=model, memory =memory, optimizer = optimizer)

# %%[markdown]

# After instantiated the client, we can now add the event handler functions. These functions are executed when we receive a certain type of message.

# We first define the function that will handle our received messages. 
# Such a function requires the "msg" as first parameter, which contains the all message information as dictionary.
# Additional we can add arguments for the other objects we use in the functions (e.g. model and memory). The naming needs to be the same as the kwargs of the client.

# The forward function of our component will receive the wind speed and the target information in the "DATA" entry of the message.
# It will convert the received data to a pytorch tensor and run it through the defined neural network model.
# The result  is converted back to a list and send per message in the "DATA" part.
# Additional we store the received data in our memory dictionary with the key based on the reply message id. Once we get a reply to this message with the gradient information we can retrieve the original data.

# Finally we add a new event handler to our client object.
# This includes the defined function that is executed when the handler is active as well as the MESSAGE_TYPE and response ROOM of our response message. IN this our response will be a DATA.FORWARD type to the 'model' room.
# We also need to set the Trigger to define which incoming messages the handler needs to process. In this case we will react to messages of type DATA.FORWARD in the 'input' room.

# Once added the event handler, the client will be added to the message rooms we require. 

# %%

def torch_forward(msg,model, memory):
    x =torch.FloatTensor(msg["DATA"])
    id_from = msg["ID"]
    id = uuid.uuid4().hex
    memory[id] = {"ID_FROM": id_from, "DATA": x}
    # print(id)
    # print(x)
    y = model(x)
    return {"ID": id, "DATA": y.tolist()}

client.add_eventHandler(torch_forward,MESSAGE_TYPE.DATA.FORWARD,responseRooms='model',trigger=Trigger(MESSAGE_TYPE.DATA.FORWARD, 'input'))

# %%[markdown]

# Similar to the forward handler we also want to handle messages we receive with the gradient feedback information to update our neural network.

# Again we first define the handle function using the received message (msg), the model, the memory as well as the prior defined optimizer.
# In this case the "DATA" entry of our message will contain gradient information, which we convert to tensors and then use in the backward operation of our pytorch model.
# To calculate the backward values we retrieve the according forward data from our memory and pass them through the model. 
# We can then execute the  optimizer step for our model. 
# Finally we calculate the gradients for the input data based on the model and the received output gradients and send them per message to another component.
# Remark: In this example the input gradients are not used anymore and are not necessary, but are kept in as example.

# The event handler for our backward pass is set up to handle all DATA.GRADIENT type messages from the 'model' room and send the reply message as DATA.GRADIENT to the 'input' room.

# %%

def torch_backward(msg,model, memory, optimizer):
    msg_id = msg["ID"]
    g = torch.FloatTensor(msg["DATA"])
    if msg_id in memory:
        id = memory[msg_id]["ID_FROM"]
        x = memory[msg_id]["DATA"]
        optimizer.zero_grad()
        y = model(x)
        # print(msg_id)
        # print(x)
        y.backward(gradient=g)
        optimizer.step()
        input_gradient = torch.autograd.functional.vjp(model, x,g)[1]
        return {"ID":id,"DATA": input_gradient.tolist()}
    return None

client.add_eventHandler(torch_backward,MESSAGE_TYPE.DATA.GRADIENT,responseRooms='input',trigger=Trigger(MESSAGE_TYPE.DATA.GRADIENT, 'model'))


# %%[markdown]

# After setting up all the required logic, we finally start our client to listen to new incoming messages.

# %%

client.listen()