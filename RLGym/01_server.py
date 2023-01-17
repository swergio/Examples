# %%[markdown]

# # WebSocket Server

# This file starts the websocket server to communicate via message in a separate process. 

# We need to specify the IP and the PORT of the server as well as the message FORMAT and the HEADER_LENGTH. All of these information have to stay the same across the server and all clients.

# %%
# Import socket package to retrieve IP of host and the Server class from swergio
import socket
from swergio import Server

# Set server PORT to 8080
PORT = 8080
# Get IP of current host
SERVER = socket.gethostbyname(socket.gethostname())
print(SERVER)
# Using utf-8 as message format
FORMAT = 'utf-8'
# Message header has a maximum length of 10
HEADER_LENGTH = 10

# Initiate the Server class
server = Server(SERVER, PORT, FORMAT, HEADER_LENGTH)
# Start the server
server.start()