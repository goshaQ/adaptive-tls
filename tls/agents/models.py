from ray.rllib.models.model import Model


class AdaptiveTrafficlightModel(Model):
    def _build_layers(self, inputs, num_outputs, options):
        raise NotImplementedError

    def _build_layers_v2(self, input_dict, num_outputs, options):
        print(input_dict)
