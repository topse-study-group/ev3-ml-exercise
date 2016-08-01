#!/usr/bin/env python

import environ
import requests, json
import random
from flask import Flask, request, jsonify
from agent import Agent
from perceptron import Perceptron
from double_layer import Double

Mode = environ.default_mode
#Model = Perceptron()
Model = Double()
Controller = Agent()
LastStatus = [0.5] * environ.status_length
LastAction = 0

app = Flask(__name__)

def demo_agent(lval, rval):
    diff = lval - rval
    direction = 0 # Forward
    if diff > 0.5:
	direction = 1 # Right
    elif diff < -0.5:
        direction = 2 # Left
    if lval > 0.6 and rval > 0.6:
	direction = random.randint(0,2) # Stray...
    return direction

@app.route('/mode', methods=['POST'])
def set_mode():
    global Mode, Model
    Mode = request.json['mode']
    if request.json['reset'] == 1:
        print "Resetting the model...."
        Model.reset()

    response = jsonify({'mode': Mode})
    response.status_code = 200
    return response

@app.route('/sensor', methods=['POST'])
def receive_sensor():
    global Controller, LastStatus, LastAction
    lval, rval = request.json['sensor_l'], request.json['sensor_r']
    last_l, last_r = LastStatus[-2], LastStatus[-1]
    reward = 0
    if lval < last_l: reward += last_l - lval
    if rval < last_r: reward += last_r - rval
    new_status = LastStatus[2:] + [lval, rval]
    Controller.add_experience((LastStatus, LastAction, reward, new_status))

    direction = 0 
    if Mode == 0: # Demo
        direction = demo_agent(lval, rval)
    if Mode == 1: # Training
        direction = Controller.get_action(Model, new_status, train=True)
    if Mode == 2: # Evaluation
        direction = Controller.get_action(Model, new_status, train=False)

    LastStatus = new_status
    LastAction = direction

    response = jsonify({'direction': direction, 'score': reward})
    response.status_code = 200
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
