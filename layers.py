import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ZeroPadding2D, ReLU, MaxPooling2D


class ReflectionPadding2D(tf.keras.layers.Layer):
    # Defining a reflection pad layer for keras, copied completely from StackOverflow.
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')

    def get_config(self):
        config = super().get_config().copy()
        return config


def res_block(inputs, layer, downsample=False):
    filters = inputs.shape[-1]
    filters *= 2 if downsample else 1
    strides = 2 if downsample else 1
    pad1 = ZeroPadding2D(1)(inputs)
    name = 'en.layer' + str(layer[0]) + '.' + str(layer[1]) + '.'
    conv1 = Conv2D(filters, 3, activation='linear', use_bias=False, strides=strides,
                   name=name + 'conv1')(pad1)
    bn1 = BatchNormalization(momentum=0.9, epsilon=1e-5, name=name + 'bn1')(conv1)
    relu1 = ReLU()(bn1)
    pad2 = ZeroPadding2D(1)(relu1)
    conv2 = Conv2D(filters, 3, activation='linear', use_bias=False,
                   name=name + 'conv2')(pad2)
    bn2 = BatchNormalization(momentum=0.9, epsilon=1e-5, name=name + 'bn2')(conv2)

    if not downsample:
        add = bn2 + inputs
    else:
        name += 'downsample.'
        conv3 = Conv2D(filters, 1, activation='linear',
                       use_bias=False, strides=2, name=name + '0')(inputs)
        bn3 = BatchNormalization(momentum=0.9, epsilon=1e-5, name=name + '1')(conv3)
        add = bn2 + bn3

    relu2 = ReLU()(add)
    return relu2


def conv_block(size, inTensor, disp=False, cnt=''):
    name = 'dispconv' if disp else 'upconv'
    name = 'de.' + name + '.' + str(len("{0:b}".format(size)) - 5) + '.' + cnt
    filters = 1 if disp else size
    x = ReflectionPadding2D()(inTensor)
    x = tf.keras.layers.Conv2D(filters, 3, name=name)(x)
    if not disp:
        x = tf.keras.layers.ELU()(x)
    else:
        x = tf.keras.activations.sigmoid(x)
    return x


def up_conv(size, firstTensor, secondTensor=None):
    x = conv_block(size, firstTensor, cnt='0')
    x = tf.keras.layers.UpSampling2D()(x)
    if size > 16:
        x = tf.keras.layers.concatenate([x, secondTensor], axis=-1)
    x = conv_block(size, x, cnt='1')
    return x
