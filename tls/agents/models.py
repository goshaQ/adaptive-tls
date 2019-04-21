from ray.rllib.models.model import Model
from ray.rllib.models import ModelCatalog

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import (
    Conv2D,
    Dense,
)


class AdaptiveTrafficlightModel(Model):
    def _build_layers(self, inputs, num_outputs, options):
        raise NotImplementedError

    def _build_layers_v2(self, input_dict, num_outputs, options):

        print(input_dict['obs'])
        print(num_outputs)


def register_model():
    ModelCatalog.register_custom_model('adaptive-trafficlight', AdaptiveTrafficlightModel)
