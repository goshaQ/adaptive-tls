from environment.constants import (
    MIN_GREEN,
    YELLOW_TIME,
    SIMULATION_STEP,
)


class Trafficlight:
    def __init__(self, connection, trafficlight_id, additional):
        self.connection = connection
        self.trafficlight_id = trafficlight_id
        self.additional = additional

        self.phases = connection.trafficlight.getCompleteRedYellowGreenDefinition(trafficlight_id)[0].getPhases()
        self.complete_phases = [idx for idx, phase in enumerate(self.phases) if 'y' not in phase.state]

        self.default_program = connection.trafficlight.getProgram(trafficlight_id)
        self.next_phase = 0
        self.current_phase = 0
        self.current_phase_duration = 0
        self.prev_traffic = set()

        self.lanes = [link[0][0] for link in connection.trafficlight.getControlledLinks(trafficlight_id)]
        self.prev_queue_length = 0

    def get_throughput(self):
        r"""Computes throughput of a trafficlight based on stop bar detectors information.
        The returned number of cars that passed through intersection equal to the number
        of cars passed since the last call of the method.

        :return: throughput of a junction.
        """
        if self.additional is None:
            raise ValueError(f'The configuration doesn\'t contain entry for the trafficlight {self.trafficlight_id}')

        traffic = set()
        for segment in self.additional.values():
            for loop_id in segment[' Departure detectors ']:
                traffic.update(self.connection.inductionloop.getLastStepVehicleIDs(loop_id))

        throughput = len(traffic - self.prev_traffic)
        self.prev_traffic = traffic

        return throughput

    def get_queue_length(self):
        r"""Computes the difference of sums of queues at each line on the intersection between
        two calls of the method.

        :return: total queue length
        """
        max_queue_sum = 0
        for phase_idx in self.complete_phases:
            max_queue = 0
            for lane_id, signal in zip(self.lanes, self.phases[phase_idx].state):
                if signal.lower() == 'g':
                    queue = self.connection.lane.getLastStepHaltingNumber(lane_id)
                    if max_queue < queue:
                        max_queue = queue
            max_queue_sum += max_queue**2

        diff = self.prev_queue_length - max_queue_sum
        self.prev_queue_length = max_queue_sum

        return diff

    def update_phase(self):
        r"""Sends the signal to switch the trafficlight to the next phase.

        :return: none.
        """
        self.connection.trafficlight.setProgram(self.trafficlight_id, self.default_program)
        self.connection.trafficlight.setPhase(self.trafficlight_id, self.next_phase)
        self.connection.trafficlight.setPhaseDuration(self.trafficlight_id, SIMULATION_STEP)

        self.current_phase = self.next_phase

    def set_next_phase(self, new_phase):
        r"""Decides whether it is time to switch to the next phase based on an agent action.
        If switch happens and duration of the current phase is less than minimum time on
        a phase, nothing happens. Otherwise, the trafficlight switches to the yellow phase
        before switching to the next phase.

        :param new_phase: phase on which the trafficlight should switch next.
        :return: none.
        """
        new_phase = self.complete_phases[new_phase]

        if new_phase == self.current_phase or self.current_phase_duration < MIN_GREEN:
            self.current_phase_duration += SIMULATION_STEP
            self.next_phase = self.current_phase
        else:
            self.current_phase_duration = SIMULATION_STEP - YELLOW_TIME
            self.next_phase = new_phase
            self._set_yellow_phase()

    def _set_yellow_phase(self):
        yellow_phase_state = ''.join(
            ['y' if signal.lower() == 'g' and new_signal == 'r' else new_signal
             for signal, new_signal in zip(self.phases[self.current_phase].state, self.phases[self.next_phase].state)])

        self.connection.trafficlight.setRedYellowGreenState(self.trafficlight_id, yellow_phase_state)
        self.connection.trafficlight.setPhaseDuration(self.trafficlight_id, YELLOW_TIME)
