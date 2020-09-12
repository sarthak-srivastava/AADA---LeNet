from contextlib import ExitStack

# import tflearn
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tf_slim as slim


def adversarial_discriminator(net, layers, scope='adversary', leaky=False):
#     if leaky:
#         activation_fn = tflearn.activations.leaky_relu
#     else:
    activation_fn = tf.nn.relu
    with ExitStack() as stack:
        stack.enter_context(tf.variable_scope(scope))
        stack.enter_context(
            slim.arg_scope(
                [slim.fully_connected],
                activation_fn=activation_fn,
                weights_regularizer=slim.l2_regularizer(2.5e-5)))
        for dim in layers:
            net = slim.fully_connected(net, dim)
        net = slim.fully_connected(net, 2, activation_fn=None)
    return net
