# %%[markdown]

# # User interface

# In this script will start all other components and then define a component to enable user input to actively label new  data.
# To play the generated melodies we will use functions from the music_generator.py file which is inspired by https://github.com/kiecodes/generate-music.
# %%
import uuid
import socket

from swergio  import Client, MESSAGE_TYPE, Trigger
from swergio_toolbox.swarm_control import Swarm
from music_generator import *
# %%[markdown]

# We will use the swarm class from the swergio toolbox to simplify the handling of multiple components at once. 
# Of course we can also just start each component one by one by just running the according script.

# The swarm class requires a swarm.yaml file, that contains the specification about each component most importantly the path to each script.
# Once the YAML is defined we run each component at once by instantiating a swarm object and calling the swarm.start() method.

# %%
swarm = Swarm()
swarm.start()

# %%[markdown]

# After all the components are running, we can define our user component that will handle the labelling of new data.
# We will set up a swergio client with a name and the same information as before and also join the 'control'

# %%
COMPONENT_NAME = 'user'

PORT = 8080
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
HEADER_LENGTH = 10
# %%[markdown]

# Now let's define a few settings for playing the generate melody including the key, scale and bpm.
# We need to start a pyo server to be able to hear the sound.
# Then we can create the swergio client and add it to the 'control' room. 

# %%
num_steps = 1
pauses = True
key = 'C'
scale= 'major'
root = 4 
bpm = 128

s = Server().boot()

client = Client(COMPONENT_NAME,SERVER,PORT,FORMAT,HEADER_LENGTH)

client.join_room('control')

# %%[markdown]

# For the labeling process we will define a handler that is trigger once the 'user' gets a new message including genomes to label.
# We convert the genome to a pyo event and play it. While listening the user can give input about his rating between 0 and 5 for the melody.
# Once all examples are labeled we send back the genomes including the ratings as FITNESS.

# %%
def label(msg):
    if "GENOMES" in msg.keys():
        genomes = msg["GENOMES"]
        ratings = []
        for genome in genomes:
            events = genome_to_events(genome, NUM_BARS, NUM_NOTES, num_steps, pauses, key, scale, root, bpm)
            for e in events:
                e.play()
            s.start()

            rating = input("Rating (0-5)")

            for e in events:
                e.stop()
            s.stop()
            
            ratings.append(float(rating))
            
        return {"GENOMES": msg["GENOMES"], "FITNESS": ratings}
    
client.add_eventHandler(label,MESSAGE_TYPE.DATA.CUSTOM,responseRooms='user',trigger=Trigger(MESSAGE_TYPE.DATA.CUSTOM,'user'))

# %%[markdown]

# Finally lets create a function that will send the command to start the evolutionary process.

# %%
def start():
    msg = {'ROOT_ID':uuid.uuid4().hex, 
        'ID':uuid.uuid4().hex,
        'TYPE': MESSAGE_TYPE.DATA.CUSTOM.id, 
        'CMD': 'START',
        'TO_ROOM': 'control'
    } 
    client.send(msg)

# %%[markdown]

# We can now start the generation of music and listen to the messages when we need to provide new labels.

# %%  
start()
client.listen()

# %%[markdown]

# Finally we can stop all the components (Server,ControlModel,Trebuchet and Trainer) all  at once with  the swarm.stop() method.
# %%
swarm.stop()

# %%
