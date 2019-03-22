from tls.additional.induction_loops import CONFIGURATION


class Interaction:
    def __init__(self, simulation):
        self.simulation = simulation

    def get_trafficlight_throughput(self, trafficlight_id):
        r"""Computes throughput of a trafficlight based on stop bar detectors information.

        :param trafficlight_id: trafficlight ID.
        :return: throughput of a junction.
        """
        if trafficlight_id not in CONFIGURATION:
            raise ValueError(f'The configuration doesn\'t contain entry for the trafficlight {trafficlight_id}')

        throughput = 0
        for segment in CONFIGURATION[trafficlight_id].values():
            for loop_id in segment['stopbar_detectors']:
                throughput += self.simulation.inductionloop.getLastStepVehicleNumber(loop_id)
        return throughput
