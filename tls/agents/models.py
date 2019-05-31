from ray.rllib.models.model import Model
from ray.rllib.models import ModelCatalog

import tensorflow as tf


class AdaptiveTrafficlightModel(Model):
    def _build_layers(self, inputs, num_outputs, options):
        raise DeprecationWarning

    def _build_layers_v2(self, input_dict, num_outputs, options):
        is_training = input_dict['is_training']
        inputs = input_dict['obs']
        obs, action_mask = inputs['obs'], inputs['action_mask']

        # Conv1 Layer : 32x32x32
        conv1 = tf.layers.conv2d(
            inputs=obs,
            filters=32,
            kernel_size=[4, 4],
            padding='same',
            activation=tf.nn.relu
        )

        # Pool1 Layer --: 16x16x32
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2
        )

        # Conv2 Layer --: 16x16x64
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[4, 4],
            padding='same',
            activation=tf.nn.relu
        )

        # Pool2 Layer --: 8x8x64
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=2
        )

        # Dense1 Layer --: 1024
        dense = tf.layers.dense(
            inputs=tf.layers.flatten(pool2),
            units=1024,
            activation=tf.nn.relu
        )

        dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.4,
            training=is_training
        )

        # Dense2 Layer --: num_outputs
        logits = tf.layers.dense(
            inputs=dropout,
            units=num_outputs,
            activation=None
        )

        # Mask out invalid actions
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        masked_logits = inf_mask + logits

        return masked_logits, dense


def register_model():
    ModelCatalog.register_custom_model('adaptive-trafficlight', AdaptiveTrafficlightModel)
