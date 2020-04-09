import tensorflow as tf
import torch
from layers import *


# Define the model using Keras Functional API:

inputs = tf.keras.Input(shape=(192, 640, 3))
encoder = []
outputs = []

# Encoder part:
x = (inputs - 0.45) / 0.225
x = ZeroPadding2D(3)(x)
x = Conv2D(64, 7, strides=2, activation='linear', use_bias=False, name='conv1')(x)
x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
x = ReLU()(x)
encoder.append(x)
x = ZeroPadding2D(1)(x)
x = MaxPooling2D(3, 2)(x)

for i in range(1, 5):
    x = res_block(x, (i, 0), i > 1)
    x = res_block(x, (i, 1))
    encoder.append(x)


# Decoder part:
x = up_conv(256, encoder[4], encoder[3])
x = up_conv(128, x, encoder[2])
outputs.append(conv_block(128, x, disp=True))

x = up_conv(64, x, encoder[1])
outputs.append(conv_block(64, x, disp=True))

x = up_conv(32, x, encoder[0])
outputs.append(conv_block(32, x, disp=True))

x = up_conv(16, x)
outputs.append(conv_block(16, x, disp=True))

outputs = outputs[::-1]
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='depth')


# Loading the weights from the PyTorch files.
# The weights for the mono_640x192 model are can be obtained from the original PyTorch repo at
# https://github.com/nianticlabs/monodepth2

encoder_path = 'models/mono_640x192/encoder.pth'
decoder_path = 'models/mono_640x192/depth.pth'
loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
loaded_dict = torch.load(decoder_path, map_location='cpu')

model.get_layer('conv1').set_weights(
    [loaded_dict_enc['encoder.conv1.weight'].numpy().transpose(2, 3, 1, 0)])

model.get_layer('bn1').set_weights(
    [loaded_dict_enc['encoder.bn1.weight'].numpy(),
     loaded_dict_enc['encoder.bn1.bias'].numpy(),
     loaded_dict_enc['encoder.bn1.running_mean'].numpy(),
     loaded_dict_enc['encoder.bn1.running_var'].numpy()])

for layer in model.layers:
    name = layer.name.split('.')
    if name[0] == 'en':
        name = '.'.join(name[1:])
        num_weights = len(layer.get_weights())
        if num_weights == 1:
            layer.set_weights([loaded_dict_enc['encoder.' + name + '.weight']
                              .numpy().transpose(2, 3, 1, 0)])
        else:
            layer.set_weights(
                [loaded_dict_enc['encoder.' + name + '.weight'].numpy(),
                 loaded_dict_enc['encoder.' + name + '.bias'].numpy(),
                 loaded_dict_enc['encoder.' + name + '.running_mean'].numpy(),
                 loaded_dict_enc['encoder.' + name + '.running_var'].numpy()])

    if name[0] == 'de':
        if name[1] == 'upconv':
            num = str(2 * (4 - int(name[2])) + int(name[3]))
            layer.set_weights([loaded_dict['decoder.' + num + '.conv.conv.weight']
                              .numpy().transpose(2, 3, 1, 0),
                               loaded_dict['decoder.' + num + '.conv.conv.bias'].numpy()])
        else:
            num = str(int(name[2]) + 10)
            layer.set_weights([loaded_dict['decoder.' + num + '.conv.weight']
                              .numpy().transpose(2, 3, 1, 0),
                               loaded_dict['decoder.' + num + '.conv.bias'].numpy()])


# Optional:
# Save the model to a h5 file, and to a quantized TFlite model.
model.save('model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open("model.tflite", "wb").write(tflite_model)
