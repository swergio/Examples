# %%[markdown]

# # Evolutionary Algorithm

# Our idea is to improve our models not just by reinforcement learning alone, but also allow a smart selection and improvement of the best model at hand. 
# Therefore we will implement a evolutionary approach, that will evaluate which of the models we favour and based  on these models generate a new set of possible models.
# To simplify the implementation we will leverage some functionalities from the deap package.

# %%
import uuid
import socket
from deap import base, creator, tools, algorithms
import random

from swergio  import Client, Trigger, MESSAGE_TYPE

# %%[markdown]

# Again let's set the COMPONENT_NAME to evolution and then specify the IP and the PORT as well as the message FORMAT and the HEADER_LENGTH. 
# All of these information have to stay the same across the server and all clients.

# %%
COMPONENT_NAME = 'evolution'

PORT = 8080
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
HEADER_LENGTH = 10

# %%[markdown]

# Now we have to define how our evolutionary algorithm, which will select and create new populations.
# First we define the genome size, which in our case needs to be the number of weights of our policy network. The evolutionary algorithm will change the genome/network weights and there fore provide us with a new policy.
# We also define the numbers of models we'll have in our network as well as probabilities how often  the algorithm will mate or mutate our genomes.

# As we use the deap package, we will define the methods how we create and mutate/mate accordingly. In our case we simply generate a genome based on random weights and mate via the provided csTwoPoint algorithm as well as mutate with mutGaussian.
# To improve the performance the values can of course be adjusted. For more details on how to use the deap package please refer to the documentation.
# %%
GENOME_SIZE = 755
NR_OF_MODELS = 20
CXPB = 0.5
MUTPB = 0.01

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("weight_bin", random.random)   #Initiate random weights
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.weight_bin, n=GENOME_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian,mu =0, sigma=1, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=3)

# %%[markdown]

# Since we defined the basis of our evolutionary algorithm we can now define a first population including genomes for each of our model.
# We will also define to dictionaries to store the weights of each model and well as the contribution information, we will receive from the models and the aggregation component.
# Finally we create the swergio Client by passing the required settings (NAME, SERVER, PORT etc. ) as well as the prior defined objects as keyword arguments, so they can be refereed to in our handling functions.
# %%
population = toolbox.population(n=NR_OF_MODELS)
memory_weights = {}
memory_contribution = {}

client = Client(COMPONENT_NAME,SERVER,PORT,FORMAT,HEADER_LENGTH,toolbox = toolbox,population = population, memory_weights = memory_weights,memory_contribution = memory_contribution )


# %%[markdown]

# To start the evolutionary algorithm we will need the current weights of the models as well as the contribution of each model to rank them.
# We will therefor send a message containing the CMD GET in the 'evolution' room once we receive a message in 'control' with EVOLUTION true.

# %%
def get_infos(msg):
    if "EVOLUTION" in msg.keys():
        if msg["EVOLUTION"]:
            return {"CMD" :"GET"}

client.add_eventHandler(get_infos,MESSAGE_TYPE.DATA.CUSTOM,responseRooms='evolution',trigger=Trigger(MESSAGE_TYPE.DATA.CUSTOM,'control'))

# %%[markdown]
# Once we received all weights from the models, we have to convert this information into our population with the different genomes to b able to use the deap package.
# For this transformation we define a helper function, we can use in our event handler.
# This function basically just extracts the weights from the weights_dict memory and writes them into each genome of our population.
# It returns a list of clones of each individual as offspring, we can further use to mutate and mate. 

# %%
def load(population,weights_dict):
    offspring = [toolbox.clone(ind) for ind in population]

    weights_keys = list(weights_dict.keys())

    for i in range(len(offspring)):
        individual = offspring[i]
        weights = weights_dict[weights_keys[i]]
        
        for j in range(len(weights)):
            individual[j] = weights[j]
        
        offspring[i] = individual
        del offspring[i].fitness.values

    return offspring
# %%[markdown]

# We now define the handler function to gather the weights and contribution information and once everything is available to start the evolution.
# If we receive a message in the 'evolution' room with the weights or the contribution we store the infos accordingly in our memory dicts.
# When both dictionaries have all required information we start the evolution process by loading the weight to the population, evaluating each individual by contribution, selecting the best individuals and finally vary the pool of individuals to get a new generation.
# With the new generation we convert the genomes back to weights and send them back to the models to update there neural networks.
# We also send a message to the 'control' room with th information that the evolutionary step is done.

# %%
def evolution(msg,toolbox, population, memory_weights,memory_contribution ):

    ## Save Current genomes
    if "WEIGHTS" in msg.keys() and "COMPONENT_ID" in msg.keys():
        k = msg["COMPONENT_ID"]
        v = msg["WEIGHTS"]
        memory_weights[k] = v
    
    ## Save contribution
    if "CONTRIBUTION" in msg.keys():
        contr = msg["CONTRIBUTION"]
        for k,v in contr.items():
            memory_contribution[k] = v    
    
    ## IF ALL AVAILABLE DO EVOLUTION
    if len(memory_weights.keys()) == NR_OF_MODELS and len(memory_contribution.keys()) == NR_OF_MODELS:
        weights_dict = memory_weights
        ## load
        population = load(population,weights_dict)
        
        ## Evaluation
        weights_keys = list(weights_dict.keys())
        for i in range(len(population)):
            ind = population[i]
            fit = (memory_contribution[weights_keys[i]],)
            ind.fitness.values = fit
        
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)
        
        new_weights_dict = {}
        for i in range(len(offspring)):
            new_weights_dict[weights_keys[i]] = offspring[i]
    
        ## SEND FEEDBACK TO CONTROL    
        msg = {'ROOT_ID':uuid.uuid4().hex, 
            'ID':uuid.uuid4().hex,
            'TYPE': MESSAGE_TYPE.DATA.CUSTOM.id, 
            'STATUS': 'EVO_DONE',
            'TO_ROOM': 'control'
        } 
        client.send(msg)
                
        ## Send new weights
    
        return {"CMD" :"SET","WEIGHTS": new_weights_dict}

client.add_eventHandler(evolution,MESSAGE_TYPE.DATA.CUSTOM,responseRooms='evolution',trigger=Trigger(MESSAGE_TYPE.DATA.CUSTOM,'evolution'))

# %%[markdown]

# After setting up all the required logic, we finally start our client to listen to new incoming messages.

# %%
client.listen()