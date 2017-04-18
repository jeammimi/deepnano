from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras import backend as K
from keras.layers.core import Reshape, Lambda
from keras.optimizers import SGD, RMSprop
from keras.layers.merge import Concatenate


inputs = Input(shape=(40, 4))
Nbases = 5 + 1
size = 20

l1 = Bidirectional(LSTM(size, return_sequences=True), merge_mode='concat')(inputs)
l2 = Bidirectional(LSTM(size, return_sequences=True), merge_mode='concat')(l1)
l3 = Bidirectional(LSTM(size, return_sequences=True), merge_mode='concat')(l2)
out_layer1 = TimeDistributed(Dense(Nbases, activation="softmax"), name="out_layer1")(l3)

try:
    model = Model(input=inputs, output=out_layer1)
except:
    model = Model(inputs=inputs, outputs=out_layer1)

model.compile(optimizer='adadelta', loss='categorical_crossentropy', sample_weight_mode='temporal')


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length, ctc_merge_repeated=False)


labels = Input(name='the_labels', shape=[40], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')


loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
    [out_layer1, labels, input_length, label_length])

model2 = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
#rms = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0, clipvalue=0.05)
model2.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
