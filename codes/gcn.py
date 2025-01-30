'''
Author: Yanzhe Zhang yanzhe_zhang@qq.com
Date: 2024-03-12 18:29:42
LastEditors: Yanzhe Zhang yanzhe_zhang@qq.com
LastEditTime: 2024-03-12 18:51:44
FilePath: /Part III Project/codes/gcn.py
Description: GCN layer
'''
import tensorflow as tf
from keras import activations, regularizers, constraints, initializers, layers

spdot = tf.sparse.sparse_dense_matmul
dot = tf.matmul

class GCN(layers.Layer):
    def __init__(
        self, 
        units, 
        activation=lambda x: x,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer='zeros',
        bias_regularizer=None,
        bias_constraint=None,
        activity_regularizer=None,
        **kwargs
    ):
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(GCN, self).__init__()

    def build(self, input_shape):
        feature_dim = input_shape[1][1]

        if not hasattr(self, "weight"):
            self.weight = self.add_weight(
                name="weight", 
                shape=(feature_dim, self.units),
                initializer=self.kernel_initializer,
                constraint=self.kernel_constraint,
                trainable=True
            )

        if self.use_bias:
            if not hasattr(self, 'bias'):
                self.bias = self.add_weight(
                    name="bias",
                    shape=(self.units, ),
                    initializer=self.bias_initializer,
                    constraint=self.bias_constraint,
                    trainable=True
                )

        super(GCN, self).build(input_shape)

    def call(self, inputs):
        self.A = inputs[0]
        self.X = inputs[1]

        if isinstance(self.X, tf.SparseTensor):
            h = spdot(self.X, self.weight)
        else:
            h = dot(self.X, self.weight)

        output = spdot(self.A, h)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        if self.activation:
            output = self.activation(output)
        
        return output