from additional.induction_loops import CONFIGURATION

from constants import (
    MIN_GREEN,
    YELLOW_TIME,
    SIMULATION_STEP,
)


class Trafficlight:
    def __init__(self, connection, trafficlight_id):
        self.connection = connection
        self.trafficlight_id = trafficlight_id

        self.phases = connection.trafficlight.getCompleteRedYellowGreenDefinition(trafficlight_id)[0].getPhases()
        self.complete_phases = [idx for idx, phase in enumerate(self.phases) if 'y' not in phase.state]

        self.default_program = connection.trafficlight.getProgram(trafficlight_id)
        self.next_phase = 0
        self.current_phase = 0
        self.current_phase_duration = 0

    def get_throughput(self, trafficlight_id):
        r"""Computes throughput of a trafficlight based on stop bar detectors information.

        :param trafficlight_id: trafficlight ID.
        :return: throughput of a junction.
        """
        if trafficlight_id not in CONFIGURATION:
            raise ValueError(f'The configuration doesn\'t contain entry for the trafficlight {trafficlight_id}')

        throughput = 0
        for segment in CONFIGURATION[trafficlight_id].values():
            for loop_id in segment['departure_detectors']:
                throughput += self.connection.inductionloop.getLastStepVehicleNumber(loop_id)
        return throughput

    def update_phase(self):
        self.connection.trafficlight.setProgram(self.trafficlight_id, self.default_program)
        self.connection.trafficlight.setPhase(self.trafficlight_id, self.next_phase)
        self.connection.trafficlight.setPhaseDuration(self.trafficlight_id, SIMULATION_STEP)

        self.current_phase = self.next_phase

    def set_next_phase(self, new_phase):
        r"""Decides whether it is time to switch to the next phase based on an agent action.
        If switch happens and duration of the current phase is less than minimum time on
        a phase, nothing happens. Otherwise, the trafficlight switches to the yellow phase
        preceding to the next phase.

        :param new_phase: phase on which the trafficlight should switch next.
        :return: none.
        """
        new_phase = self.complete_phases[new_phase]
        self.next_phase = new_phase

        if new_phase == self.current_phase or self.current_phase_duration < MIN_GREEN:
            self.current_phase_duration += SIMULATION_STEP
        else:
            self.current_phase_duration = SIMULATION_STEP - YELLOW_TIME
            self._set_yellow_phase()

    def _set_yellow_phase(self):
        yellow_phase_state = ''.join(
            ['y' if signal.lower() == 'g' and new_signal == 'r' else new_signal
             for signal, new_signal in zip(self.phases[self.current_phase].state, self.phases[self.next_phase].state)])

        self.connection.trafficlight.setRedYellowGreenState(self.trafficlight_id, yellow_phase_state)
        self.connection.trafficlight.setPhaseDuration(self.trafficlight_id, YELLOW_TIME)
