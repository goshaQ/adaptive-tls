class Simulation:
    r"""...

    Arguments:
        connection (Connection): connection to a TraCI-Server.
    """
    def __init__(self, connection):
        self.connection = connection

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def make_simulation_step(self, until=2):
        self.connection.simulationStep(step=until)
