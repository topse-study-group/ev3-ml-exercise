import environ

import tensorflow as tf
import numpy as np
import math

class Qnet(object): # super class
    def __init__(self):
        pass
    
    def get_action_values(self, status):
        action_values = self.q.eval(session=self.sess,
                                    feed_dict={self.x:[status]})
#        print action_values[0]
        return action_values[0] 
    
    def update_model(self, experiences):
        gamma = 0.9
        if len(experiences) < environ.batch_num:
            return
        batch_index = list(np.random.randint(0, len(experiences),
                                             environ.batch_num))
        batch = [experiences[i] for i in batch_index]
        xs = [item[0] for item in batch]
        targets = self.q.eval(session=self.sess, feed_dict={self.x:xs})

        for i, experience in enumerate(batch):
            status, action, reward, new_status = experience
            targets[i, action] = reward + gamma * np.max(self.get_action_values(new_status))
        self.sess.run(self.train_step, feed_dict={self.x:xs, self.y_:targets})
        return

###        return self.loss.eval(session = self.sess, feed_dict={self.x:xs, self.y_:targets})

