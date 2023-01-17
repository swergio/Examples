# %%[markdown]

# # Evaluation Network

# Instead of needing to evaluate and label each melody we are generating manually, our idea is to use a neural network that will take the genome as input and provide us with a reasonable evaluation.
# This component contains this network. It will provide the fitness value to the evolutionary algorithm and we will train it with our active learner component.

# %%
import torch 
from torch import nn
import uuid
import socket
from swergio  import Client, Trigger, MESSAGE_TYPE

from music_generator import NUM_BARS , NUM_NOTES, BITS_PER_NOTE
# %%[markdown]

# First  let's set the COMPONENT_NAME to evalnet and specify the IP and the PORT as well as the message FORMAT and the HEADER_LENGTH. 
# All of these information have to stay the same across the server and all clients.

# We also  define the GENOME_SIZE in the same way  as in our evolutionary algorithm to us as input dimension of our neural net.

# %%
COMPONENT_NAME = 'evalnet'

PORT = 8080
SERVER = socket.gethostbyname(socket.gethostname())
FORMAT = 'utf-8'
HEADER_LENGTH = 10

GENOME_SIZE = NUM_BARS * NUM_NOTES * BITS_PER_NOTE

# %%[markdown]

# We now define the neural network structure. The network will take genome as input and return logits for our 6 rating classes (0-5).

# %%
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(GENOME_SIZE, GENOME_SIZE*2),
            nn.ReLU(),
            nn.Linear(GENOME_SIZE*2, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 6)
        )

    # input wind, target 
    def forward(self, x):
        logits = self.layers(x)
        return logits
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

# To provide the fitness values (in our case a rating between 0 and 5) to the evolutionary algorithm we define the following handler.
# When we receive a message in the 'evolution' room containing the GENOMES but no FITNESS we pass the genomes through the neural net.
# Based on the logits we calculate the fitness per genome via argmax across all 6 classes and also the probability of each class via softmax.
# The probability we will later use in our active learning component, but for now it's enough to just send both information as message back to the 'evolution' room. 
# %%
def inference(msg,model):
    if "GENOMES" in msg.keys() and "FITNESS" not in msg.keys():
        x = torch.FloatTensor(msg['GENOMES'])
        y = model(x)
        
        fitness = torch.argmax(y, dim = -1)
        probs = torch.softmax(y,dim= -1)
        # print(fitness)
        # print(probs)
        return {"GENOMES": msg["GENOMES"], "FITNESS":  fitness.tolist(), "PROBS": probs.tolist()}
    
client.add_eventHandler(inference,MESSAGE_TYPE.DATA.CUSTOM,responseRooms='evolution',trigger=Trigger(MESSAGE_TYPE.DATA.CUSTOM, 'evolution'))
# %%[markdown]

# Since we want to be able to train our model constantly, we will also implement the handler for such a training loop.
# First let's define the forward pass we will take when we receive training data.
# For this we will take the data from the 'traininput' room pass them through our network and sending back the results to 'trainoutput' while storing the received dat in our internal memory.

# %%
def torch_forward(msg,model, memory):
    print("FORWARD")
    x =torch.FloatTensor(msg["DATA"])
    id_from = msg["ID"]
    id = uuid.uuid4().hex
    memory[id] = {"ID_FROM": id_from, "DATA": x}
    y = model(x)
    return {"ID": id, "DATA": y.tolist()}

client.add_eventHandler(torch_forward,MESSAGE_TYPE.DATA.FORWARD,responseRooms='trainoutput',trigger=Trigger(MESSAGE_TYPE.DATA.FORWARD, 'traininput'))
# %%[markdown]

# For the backward path in our training loop we define another handler.
# It will retrieve the gradient information from the 'trainoutput' room and together with the stored original data we can perform an optimizer step on our pytorch model.

# %%
def torch_backward(msg,model, memory, optimizer):
    print("BACKWARD")
    msg_id = msg["ID"]
    g = torch.FloatTensor(msg["DATA"])
    if msg_id in memory:
        id = memory[msg_id]["ID_FROM"]
        x = memory[msg_id]["DATA"]
        optimizer.zero_grad()
        y = model(x)
        y.backward(gradient=g)
        optimizer.step()
        input_gradient = torch.autograd.functional.vjp(model, x,g)[1]
        return {"ID":id,"DATA": input_gradient.tolist()}
    return None

client.add_eventHandler(torch_backward,MESSAGE_TYPE.DATA.GRADIENT,responseRooms='traininput',trigger=Trigger(MESSAGE_TYPE.DATA.GRADIENT, 'trainoutput'))

# %%[markdown]

# After setting up all the required logic, we finally start our client to listen to new incoming messages.

# %%
client.listen()