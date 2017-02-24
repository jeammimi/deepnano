import numpy as np
import pickle

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class OutLayer:
  def __init__(self):
    self.n_params = 2
    self.params = [None, None]

  def calc(self, input):
    otmp = np.dot(input, self.params[0]) + self.params[1]
    e_x = np.exp(otmp - otmp.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

class SimpleLayer:
  def __init__(self):
    self.n_params = 10
    self.params = [None for i in range(10)]
  
  def calc(self, input):
    state = self.params[9]
#    output = []
    output = np.zeros((len(input), self.params[2].shape[0]), dtype=np.float32)
    for i in range(len(input)):
      update_gate = sigmoid(np.dot(state, self.params[6]) + 
                            np.dot(input[i], self.params[4]) +
                            self.params[8])
      reset_gate = sigmoid(np.dot(state, self.params[5]) + 
                           np.dot(input[i], self.params[3]) +
                           self.params[7])
      new_val = np.tanh(np.dot(input[i], self.params[0]) + 
                        reset_gate * np.dot(state, self.params[1]) +
                        self.params[2])
      state = update_gate * state + (1 - update_gate) * new_val
      output[i] = state
    return np.array(output)

class BiSimpleLayer:
  def __init__(self):
    self.fwd = SimpleLayer()
    self.bwd = SimpleLayer()

  def calc(self, input):
    return np.concatenate([self.fwd.calc(input), self.bwd.calc(input[::-1])[::-1]],
                          axis=1)

class Rnn:
  def __init__(self):
    pass

  def predict(self, input):
    l1 = self.layer1.calc(input)
    l2 = self.layer2.calc(l1)
    l3 = self.layer3.calc(l2)
    return self.output1.calc(l3), self.output2.calc(l3)

  def debug(self, input):
    l1 = self.layer1.calc(input)
    l2 = self.layer2.calc(l1)
    l3 = self.layer3.calc(l2)
    return l1, l2, l3

  def load(self, fn):
    with open(fn, "rb") as f:
      self.layer1 = BiSimpleLayer()
      for i in range(10):
        self.layer1.fwd.params[i] = pickle.load(f)
      for i in range(10):
        self.layer1.bwd.params[i] = pickle.load(f)
      self.layer2 = BiSimpleLayer()
      for i in range(10):
        self.layer2.fwd.params[i] = pickle.load(f)
      for i in range(10):
        self.layer2.bwd.params[i] = pickle.load(f)
      self.layer3 = BiSimpleLayer()
      for i in range(10):
        self.layer3.fwd.params[i] = pickle.load(f)
      for i in range(10):
        self.layer3.bwd.params[i] = pickle.load(f)
      self.output1 = OutLayer()
      self.output2 = OutLayer()
      for i in range(2):
        self.output1.params[i] = pickle.load(f)
      for i in range(2):
        self.output2.params[i] = pickle.load(f)
