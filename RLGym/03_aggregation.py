# %%[markdown]

# # Aggregation

# Our idea is to run multiple RL models in parallel, while only using one combined action to perform in the environment.
# This component will  aggregate the proposed action from each model, so that we have one action to take. 
# To do so we will create a distribution based on the sum of each individual action distribution and the either draw our action or take the argmax in the deterministic case.

# This component will also track how much contribution each model had to the final action. We wan to use this information to rank our models to evolve only the ones with the highest contribution.
# In this example the contribution is simply defined as the proportion of the probability of each model to the taken action class. 

# %%
import torch 
from torch.distributions.utils import  logits_to_probs
from torch.distributions import Categorical
import socket
import copy

from swergio  import Client, Trigger, MESSAGE_TYPE

# %%[markdown]

# We set the COMPONENT_NAME to aggregation, which is use in the communication.

# We need to specify the IP and the PORT as well as the message FORMAT and the HEADER_LENGTH. 
# All of these information have to stay the same across the server and all clients.

# %%
COMPONENT_NAME = 'aggregation'

PORT = 8080
SERVER = socket.gethostbyname(socket.gethostname())
FORMAT = 'utf-8'
HEADER_LENGTH = 10
# %%[markdown]

# To be able to track all the models, we define how many models we expect in our network (e.g. 20).
# We also define to memory dictionaries. One two store the message from each model until we received all 20. The other one to store the contribution of each model.
# Lastly we create the swergio Client by passing the required settings (NAME, SERVER, PORT etc. ) as well as the prior defined objects as keyword arguments, so they can be refereed to in our handling functions.

# %%
NR_OF_MODELS = 20
memory = {}
contribution_memory = {}

client = Client(COMPONENT_NAME, SERVER,PORT,FORMAT,HEADER_LENGTH,memory=memory,contribution_memory = contribution_memory )


# %%[markdown]

# We'll first define a helper function to calculate the aggregated action as well as the contribution of each model.
# We therefore convert the action logits first into probabilities and the define a new categorical distribution based on the sum over all models.
# In case we need to provide a deterministic action we will return the argmax otherwise we just sample from this distribution.
# Lastly we calculate the contribution based on the selected action as proportion  of the single model probability to the total probability of this action. 

# %%
def get_action(dict_actions_logits, deterministic = False):

    all_probs = torch.stack([logits_to_probs(torch.tensor(v).unsqueeze(0)) for v in dict_actions_logits.values()])
    probs = all_probs.sum(0)

    dist = Categorical(probs=probs)
    if deterministic:
        action = torch.argmax(dist.probs, dim=1)
    else:
        action = dist.sample()
    action = action.numpy()

    ai = action[0]
    ai_cont = (all_probs[:,:,ai].squeeze()/probs[:,ai]).numpy()

    contribution = {k: ai_cont[i] for i,k in enumerate(dict_actions_logits.keys())}
    return action, contribution

# %%[markdown]

# Once we have defined the helper function, we can implement the event handler to aggregate the actions.
# Every time we receive a message of DATA.CUSTOM type in the 'logits' room the includes "LOGITS" information, we will add these to our memory with the COMPONENT_ID as key.
# If we collected all messages from all 20 models, we sample a aggregates action and calculate the contribution of each model.
# We add the contribution to our contribution memory and send the selected "ACTION" to the 'action' room.

# %%
def aggregate(msg,memory, contribution_memory):
    
    if "LOGITS" in msg.keys() and "COMPONENT_ID" in msg.keys(): 
        id = msg["COMPONENT_ID"]
        logits = msg["LOGITS"]
        memory[id] = logits
    if len(memory.keys()) == NR_OF_MODELS:
        action, contribution = get_action(memory, deterministic = msg['DETERMINISTIC'])
        for k,v in contribution.items():
            if k in contribution_memory.keys():
                contribution_memory[k] += float(v)
            else:
                contribution_memory[k] = float(v)
        memory.clear()
        return {"ACTION": action.tolist()}
    return None

client.add_eventHandler(aggregate,MESSAGE_TYPE.DATA.CUSTOM,responseRooms='action',trigger=Trigger(MESSAGE_TYPE.DATA.CUSTOM,'logits'))

# %%[markdown]

# In case the evolutionary algorithm requires the latest information about the contribution of each model to rank the models, we will provide the latest dictionary.
# Therefore we simply add another event handler that is trigger by a message in the 'evolution' room containing the CMD GET and returns the contribution_dict.
# Once we send this list we can clear our memory.

# %%
def send_contribution(msg, contribution_memory):
    if "CMD" in msg.keys():
        if msg["CMD"] == "GET":
            contribution = copy.deepcopy(contribution_memory)
            contribution_memory.clear()
            print(contribution)
            return {"CONTRIBUTION": contribution}

client.add_eventHandler(send_contribution,MESSAGE_TYPE.DATA.CUSTOM,responseRooms='evolution',trigger=Trigger(MESSAGE_TYPE.DATA.CUSTOM, 'evolution'))


# %%[markdown]

# After setting up all the required logic, we finally start our client to listen to new incoming messages.

# %%
client.listen()