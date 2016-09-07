#
#   eyeSwitch server in Python
#   Binds REP socket to tcp://*:5555
#

import zmq
import os, time
from pymouse import PyMouse
from pygame import mixer

m = PyMouse( )
mixer.init( )
acceptationSound = mixer.Sound( "acceptationSound.wav" )

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5559")

error_flag = True
counter = 0

while True:

    #  Wait for next request from client
    message = socket.recv()

    # reaction on client request
    # print("Received: %s" % message)

    if message == "Error":
        error_flag = True
        socket.send("Record")

    elif message == "ask":
        print "ODEBRANO PYTANIE: " + message

        if error_flag==True:
            if counter < 50:
                print "Odsylam"
                socket.send("record")
                print "Odeslano"
                counter +=1
            else:
                error_flag = False
                counter = 0
                socket.send("stop")
        else:
            socket.send("ok")                

    elif message == "press":
        print 'else -- ' + message
        
    else:
        acceptationSound.play( )
        socket.send("ok")
        # m.move(1320, 740)
        m.click(m.position()[0], m.position()[1], 1)

        #sending response to client
