# %%[markdown]

# # Gym Environment

# This file contains the logic for the gym environment, it will provide observations and rewards based on the provided actions.
# For our example we will use the "CartPole-v1" environment.

# %%
import uuid
import socket
import gym 

from swergio  import Client, Trigger, MESSAGE_TYPE
from swergio_toolbox.objects import MutableNumber, MutableBool

# %%[markdown]

# First we set the COMPONENT_NAME to environment, which is use in the communication.

# We also specify the IP and the PORT as well as the message FORMAT and the HEADER_LENGTH. 
# All of these information have to stay the same across the server and all clients.

# %%
COMPONENT_NAME = 'environment'

PORT = 8080
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
HEADER_LENGTH = 10

# %%[markdown]

# Before we create the swergio client, we will define a few mutable variable we will use in handler function.
# An episodes counter to track the number of episodes we've run through as well as the total number of episodes we want to run (run_episodes).
# To track the the score  we reach in the environment we add a score variable a well as a total score across all episodes.
# Additional we will have a flag if our actions should be deterministic and finally the gym environment itself, which in our case is the "CartPole-v1".
# We then can create the swergio Client by passing the required settings (NAME, SERVER, PORT etc. ) as well as the prior defined objects as keyword arguments, so they can be refereed to in our handling functions.

# %%
episodes_counter  = MutableNumber(0)
run_episodes = MutableNumber(0)
score  = MutableNumber(0)
totalscore  = MutableNumber(0)
deterministic = MutableBool(False)
env = gym.make("CartPole-v1")


client = Client(COMPONENT_NAME,SERVER,PORT,FORMAT,HEADER_LENGTH,episodes_counter = episodes_counter,
                run_episodes = run_episodes,
                score = score, 
                deterministic = deterministic,
                totalscore = totalscore,
                env = env)
# %%[markdown]

# To be able to start a run for a given number of episodes wie first define a message handler listening to messages from the 'control' room.
# Once we receive a message containing the "NR_OF_EPISODE" information we will reset all variables as well as the environment and start sending the first observation to the 'observation' room.
# The start message also contains the information if the actions should be deterministic.

# %%
def start(msg,episodes_counter, run_episodes, totalscore, env):
    # print(msg)
    if 'NR_OF_EPISODE' in msg.keys():
        run_episodes.set(msg['NR_OF_EPISODE'])
        episodes_counter.set(0)
        totalscore.set(0)
        deterministic.set(msg['DETERMINISTIC'])
        obs = env.reset()
        print(obs)
        return {'OBSERVATION': obs.tolist(), 'DETERMINISTIC': deterministic.value}

client.add_eventHandler(start,MESSAGE_TYPE.DATA.CUSTOM,responseRooms='observation',trigger=Trigger(MESSAGE_TYPE.DATA.CUSTOM,'control'))

# %%[markdown]

# Once the first observation is send, we expect a next action from the models. Based on the action, we will then take another step in environment, provide feedback to the models and so on.
# To define our event handler we therefore trigger a function when we receive a message in the 'action' room.
# As long as we haven't reach our total episodes limit, we will take a step in the environment and add the given reward to our score.
# In case the episode of the environment is done, we can reset the environment, increase or episode counter and adjust our total score.
# In both cases we'll send our reply message to the 'observation' room including the OBSERVATION, the REWARD, the DONE flag, the DETERMINISTIC flag as well as the actual ACTION we took.

# Once we're done with all episodes we provide a  feedback to the 'control' room including the total score.

# %%
def next_observation(msg,episodes_counter, run_episodes, score, totalscore, env):
    if episodes_counter.value < run_episodes.value: 
        action = msg['ACTION']
        obs, reward, done, info = env.step(action[0])
        score += reward
        if done:
            episodes_counter += 1 
            obs = env.reset()
            totalscore += score.value
            # print(score)
            score.set(0)
            ## SEND FEEDBACK TO CONTROL
            if episodes_counter.value == run_episodes.value:
                avg_score = float(totalscore.value) / run_episodes.value
                print(avg_score)
                fmsg = {'ROOT_ID':uuid.uuid4().hex, 
                    'ID':uuid.uuid4().hex,
                    'TYPE': MESSAGE_TYPE.DATA.CUSTOM.id, 
                    'STATUS': 'ENV_DONE',
                    'AVG_SCORE': avg_score,
                    'TO_ROOM': 'control'
                } 
                client.send(fmsg)
        
        return {'OBSERVATION':obs.tolist(),'REWARD': reward, 'DONE':done, 'DETERMINISTIC': deterministic.value,'ACTION': msg['ACTION']}
    
        
client.add_eventHandler(next_observation,MESSAGE_TYPE.DATA.CUSTOM,responseRooms='observation',trigger=Trigger(MESSAGE_TYPE.DATA.CUSTOM,'action'))

# %%[markdown]

# After setting up all the required logic, we finally start our client to listen to new incoming messages.

# %%
client.listen()