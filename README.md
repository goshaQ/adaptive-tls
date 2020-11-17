# Deep Reinforcement Learning for Adaptive Traffic Light Signal Control

This repository aims to provide a framework that can be used to easily define an environment compatible
with [OpenAI Gym](https://github.com/openai/gym) for an arbitrary road network defined in a SUMO simulation
configuration.

### How to train an agent [LINUX ONLY]

The pre-trained agents are not provided. However, you can train one by yourself within 4-8 hours,
dependinding on your hardware. Just run the following script:
```bash
python tls/agents/agent_dqn.py \
  --net-file path/to/*.net.xml \
  --config-file path/to/*.sumocfg \
  --additional-file path/to/*.det.xml \
  --num-iters 1000 \
  --checkpoint-freq 100 \
  --mode train
```

To evaluate the trained agent, replace path to the trained agent in `tls/agents/agent_dqn.py`
and all the files needed for the environment initialization in `tls/rollout.py` .
Then run the following script:
```bash
python tls/agents/agent_dqn.py \
  --mode eval
```

The repository also provides several other scripts to train APEX DQN and PPO agents in the directory `tls/agents/`.

### How it works
The simulation environment is based on an open-source microscopic traffic simulation package
[Simulation of Urban MObility (SUMO)](https://github.com/eclipse/sumo). The implementation of all RL
algorithms is taken from [RLlib: Scalable Reinforcement Learning](https://github.com/ray-project/ray).

On the diagram below the interactions between the components of the framework is shown. The Environment
component is the key one because it abstracts the interaction with the simulation and provides an
interface to initialize, step through and reset the simulation.

<p align="center">
  <img src="https://i.ibb.co/R44S9x8/FR-Comp.png">
</p>

During initialization of the Environment component the following things happen: 
1. The SUMO road network definition is pre-processed and for each traffic light
extracted an internal representation of the corresponding intersection that can
be used to observe the situation at the intersection by the reinforcement learning
agent in a convenient way;
2. Inside Controller component, from the additional file, defining the detectors
in the simulation, extracted information about installed in the road network
induction-loop detectors;
3. Created Controller component that is responsible for the interaction with the
environment and initialized with the extracted traffic light skeletons and information
about the detectors; Additionally, the following things happen
    - The controller Trafficlight is created for each traffic light;
    - The state observer Observer is created for each traffic light.

After an environment has been created and the SUMO process with the appropriate configuration files has been
started, the reinforcement learning agent can start to interact with the environment by calling the step through
function repeatedly, passing in a joint action. The actions are applied to the system, then the simulation is
progressed one step further, and the result of the simulation step is returned back to the agent.

### Intersection observation

The intersection defined in the sumo network configuration is represented as a 1 or 0 valued matrix.
An example of how an agent see the environment shown below.

<p align="center">
  <img src="https://i.ibb.co/KXz5W33/Int-Sim.png" width="40%" height="40%">
  <img src="https://i.ibb.co/GJ8S5z8/Int-Internal.png" width="40%" height="40%">
</p>

### Internal network representation

The internal representation of a road network definition is a JSON object and its schema is presented below.
The keys of the JSON object store information about the relative position of lanes, which constitute the
observed part of the road network. The intersection is split into several segments, one for each side of
the world. Each segment contains a list of lanes that are adjacent to the intersection. Because in the
SUMO network definition the lanes are separated by connections, each lane in the list is represented as
another list, where the sequence of lanes actually corresponds to a single physical lane with direction
and offset additionally specified. Sometimes segment can connect two intersections, where only one of them
must be uncontrolled, then additionally the segment may contain internal representation of a nested intersection.
The nested intersection has the same structure, however, with the specified offset from the main intersection
and unspecified segment that creates the connection.

```json
{
  "id": %JUNCTION_ID%,
  "offset": [%X_OFFSET%, %Y_OFFSET%],
  "segments": {
    "bottom": {
      "junction": {
        %NESTED_JUNCTION%
      },
      "lanes": [
        [
          [%START_POSITION%, %DIRECTION%, %LANE_ID%],
          ...
        ],
        ...
      ]
    },
    "right": {
      %ADJACENT_SEGMENT%
    },
    "top": {
      %ADJACENT_SEGMENT%
    },
    "left": {
      %ADJACENT_SEGMENT%
    }
  }
}
```
