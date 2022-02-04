# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Quantizers specific to default ternarization behavior."""

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantizers


def _WdrR(weights, alpha):
  weights_squared = tf.square(weights)
  return tf.reduce_sum((alpha - weights_squared) * weights_squared)


def WdrLoss(weight_matrix, lambda1, alpha):
  return lambda1 * _WdrR(weight_matrix, alpha)


class TernarizationWeightsQuantizer(quantizers.Quantizer):
  """Default ternarization quantizer."""

  def get_config(self):
    return {
        'lambdas': [1e-9, 1e-5, 1e-2],
        'steps': [7000, 10000],
        'alpha': 0.0,
    }

  def build(self, tensor_shape, name, layer):
    if layer.name != 'conv2d':
      return {}

    configs = self.get_config()
    self.lambda_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        values=configs['lambdas'], boundaries=configs['steps'])

    step = layer.add_weight(
        name=name + '_regularizer/step',
        initializer='zeros',
        dtype=tf.int32,
        trainable=False)

    beta = layer.add_weight(
        name=name + '_tanh/beta',
        shape=tensor_shape[-1:].as_list(),
        initializer=tf.keras.initializers.Constant(0.02),
        dtype=layer.dtype,
        trainable=True)

    return {'beta': beta, 'step': step}

  def __call__(self, inputs, training, weights, layer, **kwargs):
    if layer.name != 'conv2d':
      return inputs

    if not training:
      return inputs

    tanh_kernel = tf.tanh(inputs)
    weights['step'].assign_add(1)
    configs = self.get_config()
    layer.add_loss(lambda: WdrLoss(
        tanh_kernel,
        lambda1=self.lambda_fn(weights['step']),
        alpha=configs['alpha']))
    return tanh_kernel * weights['beta']


class TernarizationConvWeightsQuantizer(quantizers.LastValueQuantizer):
  """Quantizer for handling weights in Conv2D/DepthwiseConv2D layers."""

  def __init__(self):
    """Construct LastValueQuantizer with params specific for TFLite Convs."""

    super(TernarizationConvWeightsQuantizer, self).__init__(
        num_bits=8, per_axis=True, symmetric=True, narrow_range=True)

  def build(self, tensor_shape, name, layer):
    min_weight = layer.add_weight(
        name + '_min',
        shape=(tensor_shape[-1],),
        initializer=tf.keras.initializers.Constant(-6.0),
        trainable=False)
    max_weight = layer.add_weight(
        name + '_max',
        shape=(tensor_shape[-1],),
        initializer=tf.keras.initializers.Constant(6.0),
        trainable=False)

    return {'min_var': min_weight, 'max_var': max_weight}


class TernarizationConvTransposeWeightsQuantizer(quantizers.LastValueQuantizer):
  """Quantizer for handling weights in Conv2DTranspose layers."""

  def __init__(self):
    """Construct LastValueQuantizer with params specific for TFLite Conv2DTranpose."""

    super(TernarizationConvTransposeWeightsQuantizer, self).__init__(
        num_bits=8, per_axis=False, symmetric=True, narrow_range=True)

  def __call__(self, inputs, training, weights, **kwargs):
    outputs = tf.transpose(inputs, (0, 1, 3, 2))
    outputs = super(TernarizationConvTransposeWeightsQuantizer,
                    self).__call__(outputs, training, weights, **kwargs)
    outputs = tf.transpose(outputs, (0, 1, 3, 2))
    return outputs
