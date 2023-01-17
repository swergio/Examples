# %%[markdown]

# # Model

# This file contains the logic for reinforcement learning model using a A2C policy. 
# The model will receive the current observation from the environment and proposed distribution over the action space, which the aggregation component will use to determine the final action.
# To simplify the implementation of the RL model we use parts from the stable_baselines3 package including the model definition (A2C) and the RolloutBuffer to train the model.
# Every time the component gets new feedback regarding actual taken action and given reward, we store these values in the rollout buffer. Once the buffer is full, the policy is trained on the new information.

# Additional we enable this component to communicate and update the models weights with other swergio components. This enables us to run multiple models at the same time and easily adjust the model weights with an external component e.g. an evolutionary algorithm.

# %%
import torch 
import uuid
import socket

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import  RolloutBuffer
from stable_baselines3.common.utils import  get_schedule_fn
import gym
import numpy as np

from swergio  import Client, Trigger, MESSAGE_TYPE
from swergio_toolbox.objects import MutableBool


# %%[markdown]

# First we set the COMPONENT_NAME to model, which is use in the communication. 
# Additional we add a unique COMPONENT_ID which will be generated once the scrip is executed. This allows us to run multiple models in parallel while each having a unique identifier.

# We also need to specify the IP and the PORT as well as the message FORMAT and the HEADER_LENGTH. 
# All of these information have to stay the same across the server and all clients.

# %%
COMPONENT_NAME = 'model'
COMPONENT_ID = uuid.uuid4().hex

PORT = 8080
SERVER = socket.gethostbyname(socket.gethostname())
FORMAT = 'utf-8'
HEADER_LENGTH = 10

# %%[markdown]

# We can now define the RL model using stable_baselines3 methods.
# First we have to define the observation and action space. These values are dependent on the gym environment we are using. In this case the "CartPole-v1".
# We can then create our policy class including our desired learning rate and the architecture of the used policy network.
# Last we set up the RolloutBuffer to store the observation, action, reward information for our training. In thise example we choose the size of the buffer to be 5, which means we will train our policy every 5 steps in the environment.
# %%
# Define model
OBS_SPACE = gym.spaces.Box(
    np.array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38]),
    np.array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38]), 
    (4,), 
    np.float32)
ACT_SPACE = gym.spaces.Discrete(2)
policy = ActorCriticPolicy(observation_space=OBS_SPACE, action_space = ACT_SPACE, 
                           lr_schedule=get_schedule_fn(7e-4),
                           net_arch=[dict(pi=[16, 16], vf=[16, 16])])

rolloutbuffer = RolloutBuffer(5,observation_space= OBS_SPACE, action_space= ACT_SPACE,device="cpu")

# %%[markdown]

# Additional to the RL model we define an internal memory as dictionary. This is used to store and remember information from one step to the other, e.g. prior observation we will have to link to a reward we receive later.
# We also define a boolean variable to flag if training is enabled or not. Since we will change the value of this bool in the handler function we'll use th custom MutableBool class.
# Lastly we create the swergio Client by passing the required settings (NAME, SERVER, PORT etc. ) as well as the prior defined objects as keyword arguments, so they can be refereed to in our handling functions.

# %%
memory = {}
training_enabled = MutableBool(True)


client = Client(COMPONENT_NAME, SERVER,PORT,FORMAT,HEADER_LENGTH,policy=policy, rolloutbuffer =rolloutbuffer,memory = memory, training_enabled=training_enabled)

# %%[markdown]

# Lets first define our training function, which will update the  policy based on the history in the rollout buffer.
# For our training we will loop through each entry in the rollout buffer, calculate the policy, value and entropy loss and update our pytorch model in regard to the total loss.
# To fine-tune our training we can adjust some hyperparameter if required as the vf_coef (value loss coefficient), the ent_coef (entropy loss coefficient) or the maximum grad_norm.

# %%

def train(policy,rolloutbuffer):
    ent_coef = 0.0
    vf_coef = 0.5 
    max_grad_norm =  0.5
    
    policy.set_training_mode(True)
    for rollout_data in rolloutbuffer.get(batch_size=None):
        actions = rollout_data.actions
        actions = actions.long().flatten()

        values, log_prob, entropy = policy.evaluate_actions(rollout_data.observations, actions)
        values = values.flatten()

        advantages = rollout_data.advantages

        # Policy gradient loss
        policy_loss = -(advantages * log_prob).mean()

        # Value loss using the TD(gae_lambda) target
        value_loss = torch.nn.functional.mse_loss(rollout_data.returns, values)

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -torch.mean(-log_prob)
        else:
            entropy_loss = -torch.mean(entropy)

        loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss

        # Optimization step
        policy.optimizer.zero_grad()
        loss.backward()

        # Clip grad norm
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        policy.optimizer.step()
    policy.set_training_mode(False)
    rolloutbuffer.reset()

# %%[markdown]

# We can now add the event handler function to handle new messages from the environment. These functions are executed when we receive a certain type of message.

# We first define the function that will handle our received messages. 
# Such a function requires the "msg" as first parameter, which contains the all message information as dictionary.
# Additional we can add arguments for the other objects we use in the functions (e.g. policy,rolloutbuffer, memory). The naming needs to be the same as the kwargs of the client.

# The forward function of our component will receive the message including an OBSERVATION, the last ACTION took as well as the REWARD, a flag if the episode of the environment is DONE and  a flag if our action should be DETERMINISTIC.
# In case our roll out buffer is full, we will first train our policy with the data in the roll out buffer.
# We will then save the latest information to the rollout buffer, including th prior observation, estimated value and log_probs we stored in our memory from  the step before.
# We cam then evaluate the current observation state and determine the value and action logits of our policy.
# After saving these infos in our internal memory, we then add the logits of our action network including the unique COMPONENT_ID to message we will send to the aggregation component.


# Finally we add a new event handler to our client object.
# This includes the defined function that is executed when the handler is active as well as the MESSAGE_TYPE and response ROOM of our response message. In this case our response will be a DATA.CUSTOM type to the 'logits' room.
# We also need to set the Trigger to define which incoming messages the handler needs to process. In this case we will react to messages of type DATA.CUSTOM in the 'observation' room.

# Once added the event handler, the client will be added to the message rooms we require. 

# %%

def forward(msg,policy,rolloutbuffer, memory):
    # Train policy
    if rolloutbuffer.full:
        if training_enabled.value:
            rolloutbuffer.compute_returns_and_advantage(last_values=memory['values'], dones=msg['DONE'] if 'DONE' in msg.keys() else False)
            train(policy,rolloutbuffer)
        else:
            rolloutbuffer.reset()
    
    # Save to buffer
    if 'REWARD' in msg.keys():
        rewards = msg['REWARD']
        rolloutbuffer.add(memory['obs'], np.array(msg['ACTION']), rewards, memory['last_episode_starts'], memory['values'], memory['log_probs'])
    
    
    obs =torch.FloatTensor(msg["OBSERVATION"])

    with torch.no_grad():
        # Convert to pytorch tensor or to TensorDict
        obs_tensor = torch.as_tensor(obs)
        
        actions, values, log_probs = policy(obs_tensor.unsqueeze(0), deterministic = msg['DETERMINISTIC'])
        
        # Preprocess the observation if needed
        features = policy.extract_features(obs_tensor.unsqueeze(0))
        latent_pi, latent_vf = policy.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = policy.value_net(latent_vf)
        logits = policy.action_net(latent_pi)
    
    memory['obs'] = obs
    memory['actions'] = actions[0]
    memory['last_episode_starts'] = msg['DONE'] if 'DONE' in msg.keys() else True
    memory['values'] = values[0]
    memory['log_probs'] = log_probs[0]
    return {"LOGITS": logits[0].tolist(), "COMPONENT_ID": COMPONENT_ID, "DETERMINISTIC":msg['DETERMINISTIC']}

client.add_eventHandler(forward,MESSAGE_TYPE.DATA.CUSTOM,responseRooms='logits',trigger=Trigger(MESSAGE_TYPE.DATA.CUSTOM, 'observation'))


# %%[markdown]

# ## Weights Update

# In our example we want to be able to change the model weights no just by training the policy, but also to update the model based on an evolutionary algorithm. 
# This evolutionary component is running separately and will just communicate the weights via message.
# To enable this functionality here in our model component, we will first define two helper functions that can convert a pytorch model weights from and to a list/vector.

# The firs function model_weights_as_vector takes a pytorch model and provides the models weights as list. 
# The second function model_weights_as_dict requires a define pytorch model and the weights as list and will return a dictionary of the weights, that we can use to load into the model.
# %%

def model_weights_as_vector(model):
    weights_vector = []

    for curr_weights in model.state_dict().values():
        curr_weights = curr_weights.cpu().detach().numpy()
        vector = np.reshape(curr_weights, newshape=(curr_weights.size))
        weights_vector.extend(vector)

    return np.array(weights_vector)


def model_weights_as_dict(model, weights_vector):
    weights_dict = model.state_dict()

    start = 0
    for key in weights_dict:
        w_matrix = weights_dict[key].cpu().detach().numpy()
        layer_weights_shape = w_matrix.shape
        layer_weights_size = w_matrix.size

        layer_weights_vector = weights_vector[start:start + layer_weights_size]
        layer_weights_matrix = np.reshape(layer_weights_vector, newshape=(layer_weights_shape))
        weights_dict[key] = torch.from_numpy(layer_weights_matrix)

        start = start + layer_weights_size

    return weights_dict

# %%[markdown]

# With the helper functions we can now define the event handler for when the model receives a request to send the weights as well as when it gets new weights to update the model.

# The send_weights handler extracts the model weights and send them back including the unique COMPONENT_ID.
# While the set_weights handler updates the policy state dict with the received weights.
# Both handler are triggered by DATA.CUSTOM messages in the 'evolution' room, but one will only act with the CMD GET in the message the other with the CMD SET.

# While the model did send the current weights to the evolutionary algorithm and until it receives new  weights, we will set the training_enable variable to False.

# %%

def send_weights(msg,policy,training_enabled):
    if "CMD" in msg.keys():
        if msg["CMD"] == "GET":
            weights = model_weights_as_vector(policy)
            training_enabled.set(False)
            return {"WEIGHTS": weights.tolist(), "COMPONENT_ID": COMPONENT_ID}

client.add_eventHandler(send_weights,MESSAGE_TYPE.DATA.CUSTOM,responseRooms='evolution',trigger=Trigger(MESSAGE_TYPE.DATA.CUSTOM, 'evolution'))


def set_weights(msg,policy,training_enabled):
    if "CMD" in msg.keys():
        if msg["CMD"] == "SET":
            weights = msg["WEIGHTS"][COMPONENT_ID]
            policy.load_state_dict(model_weights_as_dict(policy, np.array(weights)))
            training_enabled.set(True)
        

client.add_eventHandler(set_weights,MESSAGE_TYPE.DATA.CUSTOM,responseRooms='evolution',trigger=Trigger(MESSAGE_TYPE.DATA.CUSTOM, 'evolution'))


# %%[markdown]

# After setting up all the required logic, we finally start our client to listen to new incoming messages.

# %%

client.listen()