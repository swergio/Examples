# %%[markdown]

# # Active learner

# Previously we defined a neural network that should be able to evaluate a melody and give a score about the quality. To train this network in a efficient way we will implement an active learning algorithm.
# This means instead of using a large amount of labeled data, we'll try to label the data in a smart way and only when we assume the new information the data will provide is sufficient.
# We can therefore look at this component in two ways, the one part where we will train the evaluation network based on a given set of labeled data and the second part the process on how we obtain more labeled data.

# %%
import socket
import uuid
import torch
from torch import nn
import random
import copy
import csv
import ast 
import math
from itertools import islice

from swergio import Client, Trigger, MESSAGE_TYPE, MODEL_STATUS
from swergio_toolbox.objects import MutableNumber, MutableBool
# %%[markdown]

#Let's as always start to set the COMPONENT_NAME and then specify the IP and the PORT as well as the message FORMAT and the HEADER_LENGTH. 
# All of these information have to stay the same across the server and all clients.

# %%
COMPONENT_NAME = 'trainer'

PORT = 8080
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
HEADER_LENGTH = 10


# %%[markdown]

# Now we have to define a couple of constant or mutable variables.
# As constant variables we will set the number of unlabeled data we want to collect before we ask for new labels and the number of data points we want to actively label as well as the file name we use to store all labeled data.
# For the training we define the epoch and bach size for each training run. 
# We will use mutable variables to track the epoch and batch number as well as the number of batches we have available in the labeled dataset and a flag if we are currently training.
# To store the unlabeled and labeled data we use dictionaries as well as a memory dictionary for our training process.

# After defining the loss function we use for our training we can finally create the swergio client.
# %%
NR_OF_UNLABELED_DATA = 200
ACTIVE_LABELS = 5
LABELED_FILE = "data.csv"
EPOCHS = 10
BATCH_SIZE = 20

epoch_counter = MutableNumber(0)
batch_counter = MutableNumber(0)
nr_of_batches = MutableNumber(0)
istraining = MutableBool(False)

dict_unlabeled = {}
dict_labeled = {}
memory = {}
l_fn = nn.CrossEntropyLoss()


client = Client(COMPONENT_NAME,SERVER,PORT,FORMAT,HEADER_LENGTH,
                dict_unlabeled = dict_unlabeled,
                dict_labeled = dict_labeled,
                epoch_counter = epoch_counter,
                batch_counter = batch_counter,
                nr_of_batches = nr_of_batches,
                istraining = istraining
                )


# %%[markdown]

# We will also define two helper functions. 
# One two read the CSV file with already labeled data an the other to be abl to split our stored labeled data dictionary into separate batches.

# %%
## LOAD LABELD DATA
with open(LABELED_FILE, 'r') as myFile:
    reader = csv.reader(myFile)
    dict_labeled = {ast.literal_eval(r[0]):int(r[1]) for r in reader}

def take(n, iterable, s = 0):
    "Return first n items of the iterable as a list"
    return list(islice(iterable,s, s + n))



# %%[markdown]
# TRAINING

# Let's first define the logic to handle the training loop including sending training data to the evaluation network and providing gradient feedback based on the actual label.
# We therefore define a helper function that generates a new massage. I will take a batch out of ou dictionary with the labeled data and then sends the data to 'traininput' while storing the label in memory.

# %%
def new_msg(dict_labeled, batch_counter):
    batch = take(min(BATCH_SIZE,len(dict_labeled.keys()) - batch_counter.value*BATCH_SIZE),dict_labeled.items(),batch_counter.value*BATCH_SIZE)
    data = [list(b[0]) for b in batch]
    target = [b[1] for b in batch]
    
    data_id = uuid.uuid4().hex
    d = {'ROOT_ID':data_id, 
        'ID':uuid.uuid4().hex,
        'TYPE': MESSAGE_TYPE.DATA.FORWARD.id, 
        'DATA': data, 
        'TO_ROOM': 'traininput',
        'MODEL_STATUS': MODEL_STATUS.TRAIN.id
    } 
    memory[data_id] = target
    return d

# %%[markdown]

# To send new batches of training data we set up a handler that will trigger when we receive the final gradient from the prior step in the 'traininput' room and then send a new batch. 
# We will keep track of the current batch and epoch until we reach our epoch goal.
# %%
def new_data(msg,dict_unlabeled, dict_labeled, batch_counter, nr_of_batches, epoch_counter, istraining):
    if epoch_counter.value <  EPOCHS:
        batch_counter += 1
        if batch_counter.value< nr_of_batches.value:
            d = new_msg(dict_labeled, batch_counter)
            return  d
        else:
            batch_counter.set(0)
            epoch_counter+= 1
            d = new_msg(dict_labeled, batch_counter)
            return  d
    else:
        dict_unlabeled.clear()
        istraining.set(False)
        return None        

client.add_eventHandler(new_data,MESSAGE_TYPE.DATA.FORWARD,responseRooms='traininput',trigger=Trigger(MESSAGE_TYPE.DATA.GRADIENT,'traininput'))

# %%[markdown]

# To provide the gradient feedback to the evaluation network we will add a event handler that will take the result from the 'trainoutput', calculates the loss based on the actual label w stored in our memory dict and then return the gradient in respect to the predicted label.


# %%
def backward(msg):
    pred = torch.FloatTensor(msg["DATA"])
    if msg["ROOT_ID"] not in memory.keys():
        return
    y = memory[msg["ROOT_ID"]]
    y = torch.tensor(y).long()
    memory.pop(msg["ROOT_ID"])
    l = l_fn(pred,y)
    l_gradient = torch.autograd.functional.jacobian(lambda x: l_fn(x,y),pred)
    l_gradient  = l_gradient.tolist()
    return {"ID":msg["ID"],"DATA":l_gradient}

client.add_eventHandler(backward,MESSAGE_TYPE.DATA.GRADIENT,responseRooms='trainoutput',trigger=Trigger(MESSAGE_TYPE.DATA.FORWARD,'trainoutput'))

# %%[markdown]
# # ACTIVE LABELING

# To perform th active labeling process we will first define a few helper functions.
# The first will calculate an uncertainty measure for our model prediction based on the class probability. 
# The second function will retrieve unlabeled data we want to label from the stored dictionary based on the uncertainty of each prediction.
# %%

def uncertainty(probs):
    return 1 - max(probs)

def get_unlabeled_for_labeling(dict_unlabeled):
    sorted_unlabeled = dict(sorted(dict_unlabeled.items(), key=lambda item: item[1]))
    n_items = take(ACTIVE_LABELS, sorted_unlabeled.items())
    return [list(n[0]) for n in n_items]

# %%[markdown]

# Now we can define the handler that will ask for new labels from the user.
# We will monitor the  message in 'evolution' and will store the results of the evaluation model in ou unlabeled data dictionary including the calculated uncertainty of the prediction.
# Once thi dictionary has the desired amount of data, we will pick the data points with highest uncertainty and send them to be labeled to the 'user'.
  
# %%
def label_data(msg, dict_unlabeled, dict_labeled, batch_counter,nr_of_batches, istraining ):
    if 'PROBS' in msg.keys():
        genomes = msg["GENOMES"]
        probs = msg["PROBS"]
        
        ## Add to unlabeled dict
        for g,p in zip(genomes,probs):
            if tuple(g) not in dict_labeled.keys():
                dict_unlabeled[tuple(g)] = uncertainty(p)
        
        print("GATHERED " + str(len(dict_unlabeled.keys())))
        ### if unlabeld full start active learning process
        if len(dict_unlabeled.keys()) >= NR_OF_UNLABELED_DATA and istraining.value == False:
            print("TRAIN")
            istraining.set(True)
            data = get_unlabeled_for_labeling(dict_unlabeled)
            return {"GENOMES": data}
        
    
client.add_eventHandler(label_data,MESSAGE_TYPE.DATA.CUSTOM,responseRooms='user',trigger=Trigger(MESSAGE_TYPE.DATA.CUSTOM,'evolution'))

# %%[markdown]

# When we receive the newly labeled data back from the user, we will start a new training loop including all labeled data we collected to this point.
# We therefore define a train function that sets the according training counter, calculates the number of available batches in our labeled dataset and send the first batch of data to the 'modelinput' room.
# We also define the handler to react on the incoming labeled data from the 'user'. The handler adds the newly labeled data to the dictionary as well as in the permanent csv file. Before starting the training, we will shuffle the labeled data. 


# %%
def train(dict_labeled, batch_counter,nr_of_batches, epoch_counter):
    # start trainning
    l = len(dict_labeled.keys())
    
    epoch_counter.set(0) 
    batch_counter.set(0)
    nr_of_batches.set(math.ceil(l/BATCH_SIZE)) 

    # SEND BATCH TO MODEL
    client.send(new_msg(dict_labeled, batch_counter))

### WHEN GETTING BACK NEW LABELD DATA START TRAINING ROUND
def start_training(msg,dict_unlabeled, dict_labeled, batch_counter,nr_of_batches, epoch_counter):
    if "FITNESS" in msg.keys():
        genomes = msg["GENOMES"]
        fitness = msg["FITNESS"]
        for k,v in zip(genomes,fitness):
            dict_labeled[tuple(k)] = v
        # add to data CSV
        with open(LABELED_FILE, 'a') as myFile:
            writer = csv.writer(myFile)
            [writer.writerow((tuple(k),int(v))) for k,v in zip(genomes,fitness)]
        
        random_dict_labeled = copy.deepcopy(dict_labeled)
        keys = list(random_dict_labeled.keys())
        random.shuffle(keys)
        dict_labeled.clear()
        for k in keys:
            dict_labeled[k] = random_dict_labeled[k]
    
        train(dict_labeled, batch_counter,nr_of_batches, epoch_counter)

client.add_eventHandler(start_training,MESSAGE_TYPE.DATA.CUSTOM,responseRooms='user',trigger=Trigger(MESSAGE_TYPE.DATA.CUSTOM,'user'))

# %%[markdown]

# Up to this point the model is only trained once we already collected a bunch of data and labeled it. 
# To speed up the initial training process, we will pre-train our model based on the given data in the csv file. 

# %%
# Pre-train if labeled data avaialable
if  len(dict_labeled.keys()) > 0:
    print("PRE-TRAIN")
    train(dict_labeled, batch_counter,nr_of_batches, epoch_counter)

# %%[markdown]

# After setting up all the required logic, we finally start our client to listen to new incoming messages.

# %%
client.listen()