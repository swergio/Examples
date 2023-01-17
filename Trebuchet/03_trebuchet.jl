#=
This julia script includes the logic of the trebuchet. 
We will use the Trebuchet package which includes the physical implementation a trebuchet and enables us to simulate it.
Additional we are able to calculate the gradient of the trebuchet by using the Zygote package.

To communicate with the other components, which are written in python we use the swergio package that enables us to send messages via websocket.
=#
using swergio

using Trebuchet
using Flux
using Zygote: gradient, forwarddiff
using LinearAlgebra
using Sockets
using UUIDs

#=
We set the COMPONENT_NAME to trebuchet, which is use in the communication.

We need to specify the IP and the PORT as well as the message HEADER_LENGTH. 
All of these information have to stay the same across the server and all clients.

=#
COMPONENT_NAME = "trebuchet"

SERVER = ip"127.0.1.1"
PORT = 8080
HEADER_LENGTH = 10

#=
Now we define a wraper of the shoot function of the Trebuchet package.

Our function will take the wind as parameter.
As well as logits for the angle of the trebuchet as well as the counter weight.
These values are transformed to more resonable values that the Trebuchet package can handle.

The function will return our achieved target.
=#
function shoot(wind, angle, weight)
    angle = σ(angle)*90
    weight = weight + 200
    Trebuchet.shoot((wind, Trebuchet.deg2rad(angle), weight))[2]
end
shoot(ps) = forwarddiff(p -> shoot(p...), ps)

#=
After we defined a couple of variable and functions we can instatiate the swergio client by provding the required settings.

We also instatiate a memory dictionary to store the data we recieved on the forward path to refere to them in our backward path.
=#

client = swergio.ClientModule.Client(COMPONENT_NAME,SERVER,PORT; header_length = HEADER_LENGTH)

memory = Dict()


#=
After instantiated the client, we can now add the event handler functions. These functions are executed when we receive a certain type of message.

We first define the function that will handle our received messages. 
Such a function requires the "msg" as first parameter, which contains the all message information as dictionary.


The forward function of our component will receive the wind speed, the trebuchet angle and the counter weight in the "DATA" entry of the message.
We convert this data into a format our shoot function can handle and then return the result in our new message in the "DATA" entry.
Meanwhile we alsoe store the received data in our memory dictionary with the key based on the reply message id. Once we get a reply to this message with the gradient information we can retrieve the original data.

Finally we add a new event handler to our client object.
This includes the defined function that is executed when the handler is active as well as the MESSAGE_TYPE and response ROOM of our response message. 
In this case our response will be a DATA.FORWARD type to the 'output' room.
We also need to set the Trigger to define which incoming messages the handler needs to process. 
In this case we will react to messages of type DATA.FORWARD in the 'model' room.

Once added the event handler, the client will be added to the message rooms we require. 
=#

function forward(msg)
    x = msg["DATA"]
    x = [Float64.(xx) for xx in x]
    id_from = msg["ID"]
    id = string(uuid4())
    memory[id] = Dict("ID_FROM"=>id_from,"DATA"=>x)
    y = shoot.(x)
    return Dict{String,Any}("ID"=> id,"DATA"=>y)

end

swergio.ClientModule.add_eventHandler(client,
    forward,
    swergio.messagetype.MESSAGE_TYPE.DATA.FORWARD,
    responseRooms = ["output"],
    trigger = swergio.ClientModule.Trigger(swergio.messagetype.MESSAGE_TYPE.DATA.FORWARD,"model")
)


#=
Similar to the forward handler we also want to handle messages we receive with the gradient feedback information to further pass down the gradient information to the control network.

We first define the handle function using the received message (msg).
In this case the "DATA" entry of our message will contain gradient information.

To get the gradient of our shoot function we retrieve the according forward data from our memory. 
With the gradient information of our output we can then calculate the gradient of the trebuchet shoot function in respect to the input data.
These innformation we send down to the control model via message. 
    
The event handler for our backward pass is set up to handle all DATA.GRADIENT type messages from the 'output' room and send the reply message as DATA.GRADIENT to the 'model' room.
=#
function backward(msg)
    msg_id = msg["ID"]
    g = msg["DATA"]
    if msg_id in keys(memory)
        id = memory[msg_id]["ID_FROM"]
        x = copy(memory[msg_id]["DATA"])
        # get input gradient
        J = [gradient(shoot,xx)[1] for xx in x]
        input_gradient = [gg[1]*gg[2] for gg in zip(g,J)]
        return Dict{String,Any}("ID"=>id,"DATA"=>input_gradient)
    end
end

swergio.ClientModule.add_eventHandler(client,
backward,
    swergio.messagetype.MESSAGE_TYPE.DATA.GRADIENT,
    responseRooms = ["model"],
    trigger = swergio.ClientModule.Trigger(swergio.messagetype.MESSAGE_TYPE.DATA.GRADIENT,"output")
)


#=
Finally we start our client to listen to new incoming messages.
=#
swergio.ClientModule.listen(client)