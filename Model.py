import tensorflow as tf
import numpy as np

#maskが機能しているかの簡単な実験
#モデルは、Conv+LSTM+Dense
class Model(tf.keras.Model):
    def __init__(self, frames, height, width, channels):
        super(Model, self).__init__()
        self.frames = frames
        self.height = height
        self.weight = width
        self.channels = channels 
        self.conv_layer   = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(3, (3, 3), activation='relu'), input_shape=(self.frames, self.height, self.width, self.channels))
        self.lstm_layer = tf.compat.v1.keras.layers.LSTM(10, time_major=False, activation='tanh')
        self.dense_layer = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, mask):
        with tf.device('/CPU:0'):
            #conv_layerに対するoutputsの形状(batchsize, frames, height, width, channels)
            outputs = self.conv_layer(inputs, mask=mask)
            #reshapeに対するoutputsの形状(batchsize, framges, height*width*channels)
            outputs = tf.reshape(outputs, [-1, self.frames, outputs.numpy().shape[2]*outputs.numpy().shape[3]*outputs.numpy().shape[4]])
            #lstm_layerに対するoutputsの形状(batchsize, frames*height*width*channels)
            outputs = self.lstm_layer(outputs, mask=mask)
            #dense_layerに対するoutputsの形状(batchsize, class)
            outputs = self.dense_layer(outputs)   
        return outputs
