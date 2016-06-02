import theano as th
import theano.tensor as T
from theano.tensor.nnet import sigmoid
import numpy as np
import pickle
from theano_toolkit import updates

def share(array, dtype=th.config.floatX, name=None):
  return th.shared(value=np.asarray(array, dtype=dtype), name=name)

class OutLayer:
  def __init__(self, input, in_size, n_classes):
    id = str(np.random.randint(0, 10000000))
    w = share(np.zeros((in_size, n_classes)), name="wout"+id)
    b = share(np.zeros(n_classes), name="bout"+id)
    eps = 0.0000001
    self.output = T.clip(T.nnet.softmax(T.dot(input, w) + b), eps, 1-eps)
    self.params = [w, b]

class SimpleLayer:
  def __init__(self, input, nin, nunits):
    id = str(np.random.randint(0, 10000000))
    wio = share(np.zeros((nin, nunits)), name="wio"+id)  # input to output
    wir = share(np.zeros((nin, nunits)), name="wir"+id)  # input to output
    wiu = share(np.zeros((nin, nunits)), name="wiu"+id)  # input to output
    woo = share(np.zeros((nunits, nunits)), name="woo"+id)  # output to output
    wou = share(np.zeros((nunits, nunits)), name="wou"+id)  # output to output
    wor = share(np.zeros((nunits, nunits)), name="wor"+id)  # output to output
    bo = share(np.zeros(nunits), name="bo"+id)
    bu = share(np.zeros(nunits), name="bu"+id)
    br = share(np.zeros(nunits), name="br"+id)
    h0 = share(np.zeros(nunits), name="h0"+id)

    def step(in_t, out_tm1):
      update_gate = sigmoid(T.dot(out_tm1, wou) + T.dot(in_t, wiu) + bu)
      reset_gate = sigmoid(T.dot(out_tm1, wor) + T.dot(in_t, wir) + br)
      new_val = T.tanh(T.dot(in_t, wio) + reset_gate * T.dot(out_tm1, woo) + bo)
      return update_gate * out_tm1 + (1 - update_gate) * new_val
    
    self.output, _ = th.scan(
      step, sequences=[input],
      outputs_info=[h0])

    self.params = [wio, woo, bo, wir, wiu, wor, wou, br, bu, h0]

class BiSimpleLayer():
  def __init__(self, input, nin, nunits):
    fwd = SimpleLayer(input, nin, nunits)
    bwd = SimpleLayer(input[::-1], nin, nunits)
    self.params = fwd.params + bwd.params
    self.output = T.concatenate([fwd.output, bwd.output[::-1]], axis=1)

class Rnn:
  def __init__(self, filename):
    package = np.load(filename)
    assert(len(package.files) % 20 == 4)
    n_layers = len(package.files) / 20

    self.params = []
    self.input = T.fmatrix()
    last_output = self.input
    last_size = package['arr_0'].shape[0]
    hidden_size = package['arr_0'].shape[1]
    par_index = 0
    for i in range(n_layers):
      layer = BiSimpleLayer(last_output, last_size, hidden_size)
      self.params += layer.params
      for i in range(20):
        layer.params[i].set_value(package['arr_%d' % par_index])
        par_index += 1

      last_output = layer.output
      last_size = 2*hidden_size
    out_layer1 = OutLayer(last_output, last_size, 5)
    for i in range(2):
      out_layer1.params[i].set_value(package['arr_%d' % par_index])
      par_index += 1
    out_layer2 = OutLayer(last_output, last_size, 5)
    for i in range(2):
      out_layer2.params[i].set_value(package['arr_%d' % par_index])
      par_index += 1
    output1 = out_layer1.output
    output2 = out_layer2.output
    self.params += out_layer1.params
    self.params += out_layer2.params

    self.predict = th.function(inputs=[self.input], outputs=[output1, output2])
    self.tester = th.function(inputs=[self.input], outputs=[output1, output2])

    self.lr = T.fscalar()
    self.targets = T.ivector()
    self.targets2 = T.ivector()
    self.cost = 0
    self.cost = -T.mean(T.log(output1)[T.arange(self.targets.shape[0]), self.targets]) 
    self.cost += -T.mean(T.log(output2)[T.arange(self.targets2.shape[0]), self.targets2]) 

    self.trainer = th.function(
        inputs=[self.input, self.targets, self.targets2, self.lr],
        outputs=[self.cost, output1, output2],
        updates=updates.momentum(self.params, (T.grad(self.cost, self.params)),
                                 learning_rate=self.lr, mu=0.8))

  def save(self, fn):
    pp = [p.get_value() for p in self.params] 
    np.savez(fn, *pp)
