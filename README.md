# Proximal Policy Optimization (PPO) in Tensorflow.js

`ppo-tfjs` is an open-source implementation of the Proximal Policy Optimization (PPO) algorithm using Tensorflow.js. It's a one-file script that can be loaded directly into a browser or used in a Node.js environment.

## Installation

```bash
npm install ppo-tfjs
```

## Loading

### Browser
```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
<script src="https://cdn.jsdelivr.net/npm/ppo-tfjs"></script>
```

### Node.js
```javascript
const tf = require('@tensorflow/tfjs-node-gpu')
const PPO = require('ppo-tfjs')
```

## Usage

### Create environment
`ppo-tfjs` require an environment that mimics Python's `gym` specification. The environment must have the following methods:
- `step(action)` - returns an object with the following properties:
  - `observation` - the current state of the environment
  - `reward` - the reward for the current step
  - `done` - a boolean indicating if the episode is over
- `reset()` - resets the environment and returns the initial state
The environment also defines the following properties:
- `actionSpace` - an object with the following properties:
  - `class` - the class of the action space (`Box` or `Discrete`)
  - `shape` (Box) - the shape of the action space
  - `low` (Box) - the lower bound of the action space
  - `high` (Box) - the upper bound of the action space
  - `n` (Discrete) - the number of actions in the action space
  - `dtype` - the data type of the action space (default: `float32` for `Box` and `int32` for `Discrete`)
- `observationSpace` - an object with the following properties:
  - `shape` - the shape of the observation space
  - `dtype` - the data type of the observation space (default: `float32`)

Example:
```javascript
/*
Following environment creates an agent and a goal both represented as x,y coordinates.
The agent receives rewards based on the distance to the goal (it's more like penalty here)
After each reset() the agent and goal are randomly placed in the environment.
*/
class Env {
    constructor() {
        this.actionSpace = {
            'class': 'Box',
            'shape': [2],
            'low': [-1, -1],
            'high': [1, 1],
        }
        this.observationSpace = {
            'class': 'Box',
            'shape': [4],
            'dtype': 'float32'
        }
    }
    async step(action) {
        const oldAgent = this.agent.slice(0)
        this.agent[0] += action[0] * 0.05
        this.agent[1] += action[1] * 0.05
        this.i += 1
        var reward = -Math.sqrt(
            (this.agent[0] - this.goal[0]) * (this.agent[0] - this.goal[0]) + 
            (this.agent[1] - this.goal[1]) * (this.agent[1] - this.goal[1])
        )
        var done = this.i > 30 || reward > -0.01
        return [
            [this.agent[0], this.agent[1], this.goal[0], this.goal[1]],
            reward, 
            done
        ]
    }
    reset() {
        this.agent = [Math.random(), Math.random()]
        this.goal = [Math.random(), Math.random()]
        this.i = 0
        return  [this.agent[0], this.agent[1], this.goal[0], this.goal[1]]
    }
}
const env = new Env()
```
 
### Initialize PPO and start training
```javascript
const ppo = new PPO(env, {'nSteps': 1024, 'nEpochs': 50, 'verbose': 1})
;(async () => {
    await ppo.learn({
        'totalTimesteps': 100000,
        'callback': {
            'onTrainingStart': function (p) {
                console.log(p.config)
            }
        }
    })
})()
```

## Full configuration
```javascript
const config = {
    nSteps: 512,                 // Number of steps to collect rollouts
    nEpochs: 10,                 // Number of epochs for training the policy and value networks
    policyLearningRate: 1e-3,    // Learning rate for the policy network
    valueLearningRate: 1e-3,     // Learning rate for the value network
    clipRatio: 0.2,              // PPO clipping ratio for the objective function
    targetKL: 0.01,              // Target KL divergence for early stopping during policy optimization
    netArch: {
        'pi': [32, 32],          // Network architecture for the policy network
        'vf': [32, 32]           // Network architecture for the value network
    },
    activation: 'relu',          // Activation function to be used in both policy and value networks
    verbose: 0                   // Verbosity level (0 for no logging, 1 for logging)
}
const ppo = new PPO(env, config)
```