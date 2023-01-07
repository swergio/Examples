# %%[markdown]

# # Universe

# This file contains the logic for our "universe" component. It will send the first "Hello World!" message and afterwards receives replies from the "world" component via the "galactic_chat" room and send its responses.

# %%
import uuid
import socket
from swergio  import Client, Trigger, MESSAGE_TYPE

# %%[markdown]

# We set a COMPONENT_NAME to universe, which is use in the communication.

# We need to specify the IP and the PORT as well as the message FORMAT and the HEADER_LENGTH. 
# All of these information have to stay the same across the server and all clients.

# %%
COMPONENT_NAME = 'universe'

PORT = 8080
SERVER = socket.gethostbyname(socket.gethostname())
FORMAT = 'utf-8'
HEADER_LENGTH = 10

# %%[markdown]

# We create the swergio Client by passing the required settings (NAME, SERVER, PORT etc. ).

# %%

client = Client(COMPONENT_NAME, SERVER,PORT,FORMAT,HEADER_LENGTH)

# %%[markdown]

# Similar to the "world" component we define an event handler to respond to received messages. In this case we wait until we get a message in the "galactic_chat" room and then send a new "Hello World!" message.

# %%

def reply(msg):
    print(msg)
    id = uuid.uuid4().hex
    return {"ID": id, "DATA": "Hello World!"}

client.add_eventHandler(reply,MESSAGE_TYPE.DATA.TEXT,responseRooms='galactic_chat',trigger=Trigger(MESSAGE_TYPE.DATA.TEXT, 'galactic_chat'))

# %%[markdown]

# Since the event handlers are only reacting t incoming message, on of the components should send the first message.
# We define our first "Hello World!" message by including a new message id, the message type, the room we want the message to send to as well as the message data.
# Our "universe" will then send the first message and the "world" can respond.
# %%
msg = {
    'ID': uuid.uuid4().hex, 
    'TYPE': MESSAGE_TYPE.DATA.TEXT.id, 
    'TO_ROOM': 'galactic_chat',
    'DATA': "Hello World!"}
client.send(msg)

# %%[markdown]

# After setting up all the required logic, we finally start our client to listen to new incoming messages.
client.listen()