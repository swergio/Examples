# %%[markdown]

# # Evolutionary Algorithm

# In this component we will try to generate new melodies using an evolutionary algorithm. The algorithm will generate a population where each individual contains a genome that represents a melody.
# Based on the feedback which melodies in the population are the best, we will try to improve the best melodies by mating and mutating the genomes.
# To simplify th implementation of the evolutionary algorithm we will use the deap package, that allows us to define the selection, mating and mutating process easily.

# %%
import socket
from deap import base, creator, tools, algorithms
import random
import time

from swergio  import Client, Trigger, MESSAGE_TYPE
from swergio_toolbox.objects import MutableBool 

from music_generator import NUM_BARS , NUM_NOTES, BITS_PER_NOTE

# %%[markdown]

# First we set the COMPONENT_NAME to evolution and then specify the IP and the PORT as well as the message FORMAT and the HEADER_LENGTH. 
# All of these information have to stay the same across the server and all clients.

# %%
COMPONENT_NAME = 'evolution'

PORT = 8080
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
HEADER_LENGTH = 10

# %%[markdown]

# Now we have to define how our evolutionary algorithm that should select and create new populations.
# First we define the genome size, which in our case is the combination of the number of bars in our melody, the number of notes and the bit size per note. These variable are set in music_generator.py file.
# We also define the size of our population, the probabilities how often  the algorithm will mate or mutate our genomes as well as the time we want to wait until we generate a new population in seconds.

# As we use the deap package, we will define the methods how we create and mutate/mate accordingly. In our case we simply generate a genome based on a random list of 0s and 1s and mate via the provided csTwoPoint algorithm as well as mutate with mutFlipBit.
# To improve the performance the values can of course be adjusted. For more details on how to use the deap package please refer to the documentation.
# %%
GENOME_SIZE = NUM_BARS * NUM_NOTES * BITS_PER_NOTE
POPULATION_SIZE = 50
CXPB = 0.5
MUTPB = 0.01
SEC_SLEEP_BEFORE_NEXT_GEN = 10

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("weight_bin", lambda : random.choice([0,1]))   #Initiate random weights
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.weight_bin, n=GENOME_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=3)

# %%[markdown]

# Since we defined the basis of our evolutionary algorithm we can now define a first population including genomes for each melody.
# We will also define a flag as mutable boolean to determine if our evolution is active.
# Finally we create the swergio Client by passing the required settings (NAME, SERVER, PORT etc. ) as well as the prior defined objects as keyword arguments, so they can be refereed to in our handling functions.
# %%
population = toolbox.population(n=POPULATION_SIZE)
active = MutableBool(False)

client = Client(COMPONENT_NAME,SERVER,PORT,FORMAT,HEADER_LENGTH,toolbox = toolbox,population = population, active= active)
    
# %%[markdown]

# We can now add the event handler function to handle new messages from the environment. These functions are executed when we receive a certain type of message.

# We first define the function that will handle our received messages. 
# Such a function requires the "msg" as first parameter, which contains the all message information as dictionary.
# Additional we can add arguments for the other objects we use in the functions (e.g.population, active). The naming needs to be the same as the kwargs of the client.

# The start_stop function in our case will handle commands fro the 'control' room to start with the evolution process or respectively stop it. 
# Depending on th content of the "CMD" entry in the received message dictionary, we will either set the active flag to true and return a message to the 'evolution' room containing the GENOMES information or we set the active flag to false.

# Finally we add a new event handler to our client object.
# This includes the defined function that is executed when the handler is active as well as the MESSAGE_TYPE and response ROOM of our response message. As mentioned our response will be a DATA.CUSTOM type to the 'evolution' room.
# We also need to set the Trigger to define which incoming messages the handler needs to process. In this case we will react to messages of type DATA.CUSTOM in the 'control' room.

# Once added the event handler, the client will be added to the message rooms we require. 

# %%
def start_stop(msg,population, active):
    if "CMD" in msg.keys():
        if msg["CMD"] == "START" and active == False:
            active.set(True)
            return {"GENOMES": list([list(p) for p in population])}
            
        if msg["CMD"] == "STOP" and active.value:
            active.set(False)
            
        
client.add_eventHandler(start_stop,MESSAGE_TYPE.DATA.CUSTOM,responseRooms='evolution',trigger=Trigger(MESSAGE_TYPE.DATA.CUSTOM,'control'))

# %%[markdown]

# Once we sent the genomes to the 'evolution' room we'll have to wait until we receive the fitness score for the gnomes from the component that is able to evaluate the generated melodies.
# To handle the feedback regarding fitness and based on it evolving our population, we define an event handler as following.

# Our handle function extracts the FITNESS values from the message. We add the new fitness values to our population and then use the deap functions to select and vary the pool of individuals.
# Once we have an evolved population we will send the new GENOMES again to the 'evolution' room to be evaluated. 
# %%
def evolution(msg,toolbox, population, active):
    if "FITNESS" in msg.keys() and active.value:
        time.sleep(SEC_SLEEP_BEFORE_NEXT_GEN)
        fitness = msg["FITNESS"]    
        for i in range(len(population)):
            ind = population[i]
            fit = (fitness[i],)
            ind.fitness.values = fit
        
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)
            
        return {"GENOMES": list([list(p) for p in offspring])}

client.add_eventHandler(evolution,MESSAGE_TYPE.DATA.CUSTOM,responseRooms='evolution',trigger=Trigger(MESSAGE_TYPE.DATA.CUSTOM,'evolution'))
# %%[markdown]

# After setting up all the required logic, we finally start our client to listen to new incoming messages.

# %%

client.listen()