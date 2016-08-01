import environ
import numpy as np

class Agent:
    def __init__(self):
        self.epsilon = 0.7
        self.experiences = []

    def add_experience(self, item):
        self.experiences.append(item)
        if len(self.experiences) > environ.experience_memory:
            self.experiences = self.experiences[1:]

    def get_action(self, qnet, status, train=True):
        if train and np.random.random() < self.epsilon:
            action = np.random.choice(environ.actions)
#            qnet.update_model(self.experiences)
        else:
            action_values = qnet.get_action_values(status)
            action_index = np.argmax(action_values)
            action = environ.actions[action_index]
        if train:
            qnet.update_model(self.experiences)
        return action
