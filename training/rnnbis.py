from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed

inputs = Input(shape=(None, 4))
Nbases = 5 + 1
size = 20

l1 = Bidirectional(LSTM(size, return_sequences=True), merge_mode='concat')(inputs)
l2 = Bidirectional(LSTM(size, return_sequences=True), merge_mode='concat')(l1)
l3 = Bidirectional(LSTM(size, return_sequences=True), merge_mode='concat')(l2)
out_layer1 = TimeDistributed(Dense(Nbases, activation="softmax"),name="out_layer1")(l3)
out_layer2 = TimeDistributed(Dense(Nbases, activation="softmax"),name="out_layer2")(l3)

try:
    model = Model(input=inputs, output=[out_layer1, out_layer2])
except:
    model = Model(inputs=inputs, outputs=[out_layer1, out_layer2])

model.compile(optimizer='adadelta', loss='categorical_crossentropy',sample_weight_mode='temporal')
