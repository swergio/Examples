# %%[markdown]

# # World

# This file contains the logic for our "world" component. It will receive messages from the "universe" component via the "galactic_chat" room and replies.

# %%
import uuid
import socket
from swergio  import Client, Trigger, MESSAGE_TYPE

# %%[markdown]

# We set a COMPONENT_NAME to world, which is use in the communication.

# We need to specify the IP and the PORT as well as the message FORMAT and the HEADER_LENGTH. 
# All of these information have to stay the same across the server and all clients.

# %%
COMPONENT_NAME = 'world'

PORT = 8080
SERVER = socket.gethostbyname(socket.gethostname())
FORMAT = 'utf-8'
HEADER_LENGTH = 10

# %%[markdown]

# We create the swergio Client by passing the required settings (NAME, SERVER, PORT etc. ).

# %%

client = Client(COMPONENT_NAME, SERVER,PORT,FORMAT,HEADER_LENGTH)

# %%[markdown]

# After instantiated the client, we can now add the event handler functions. These functions are executed when we receive a certain type of message.

# We first define the function that will handle our received messages. 
# Such a function requires the "msg" as first parameter, which contains the all message information as dictionary.

# The reply function of our world component will just send back a message including the text "Hello Universe!".

# We now have to add a new event handler to our client object.
# This includes the defined function that is executed when the handler is active as well as the MESSAGE_TYPE and response ROOM of our response message. In this our response will be a DATA.TEXT type to the 'galactic_chat' room.
# We also need to set the Trigger to define which incoming messages the handler needs to process. In this case we will react to messages of type DATA.TEXT in the 'galactic_chat' room.

# Once added the event handler, the client will be added to the message rooms we require. 

# %%

def reply(msg):
    print(msg)
    id = uuid.uuid4().hex
    return {"ID": id, "DATA": "Hello Universe!"}

client.add_eventHandler(reply,MESSAGE_TYPE.DATA.TEXT,responseRooms='galactic_chat',trigger=Trigger(MESSAGE_TYPE.DATA.TEXT, 'galactic_chat'))


# %%[markdown]

# After setting up all the required logic, we finally start our client to listen to new incoming messages.
client.listen()