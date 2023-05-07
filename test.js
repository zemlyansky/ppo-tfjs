const { FunctionalCallback } = require('./callbacks.js')
const PPO = require('./ppo.js')

class EnvDiscrete {
    constructor() {
        this.actionSpace = {
            'class': 'Discrete',
            'n': 4
        }
        this.observationSpace = {
            'class': 'Box',
            'shape': [2],
            'dtype': 'float32'
        }
    }
    async step(action) {
        switch (action) {
            case 0:
                this.state[1] -= 0.01
                break
            case 1:
                this.state[1] += 0.01
                break
            case 2:
                this.state[0] -= 0.01
                break
            case 3:
                this.state[0] += 0.01
                break
        }
        this.i += 1
        var reward = -Math.sqrt(this.state[0] * this.state[0] + this.state[1] * this.state[1])
        var done = this.i > 100 || reward > -0.01
        if (reward > -0.01) {
            console.log('Goal reached:', this.state)
        }
        return [this.state.slice(0), reward, done]
    }
    reset() {
        this.state = [
            Math.random() - 0.5,
            Math.random() - 0.5,
        ]
        this.i = 0
        return this.state.slice(0)
    }
}

class EnvContinuous {
    constructor() {
        this.actionSpace = {
            'class': 'Box',
            'shape': [2],
        }
        this.observationSpace = {
            'class': 'Box',
            'shape': [2],
            'dtype': 'float32'
        }
    }
    async step(action) {
        this.state[0] += action[0] * 0.1
        this.state[1] += action[1] * 0.1
        this.i += 1
        var reward = -Math.sqrt(this.state[0] * this.state[0] + this.state[1] * this.state[1])
        var done = this.i > 100 || reward > -0.01
        if (reward > -0.01) {
            console.log('Goal reached:', this.state)
        }
        return [this.state.slice(0), reward, done]
    }
    reset() {
        this.state = [
            Math.random() - 0.5,
            Math.random() - 0.5,
        ]
        this.i = 0
        return this.state.slice(0)
    }
}

test('FunctionalCallback', async () => {
    const callback = new FunctionalCallback(function () {
        this.a = 1
        return true
    })
    callback.onStep()
    expect(callback.nCalls).toBe(1)
    expect(callback.a).toBe(1)
})

test('PPO Learn (Discrete)', async () => {
    var env = new EnvDiscrete()
    var ppo = new PPO(env, {'nSteps': 50})
    await ppo.learn({
        'totalTimesteps': 100,
        'callback': {
            'onTrainingStart': function (p) {
                console.log(p.config)
            }
        }
    })
})

test('PPO Learn (Continuos)', async () => {
    var env = new EnvContinuous()
    var ppo = new PPO(env, {'nSteps': 50})
    await ppo.learn({
        'totalTimesteps': 100,
        'callback': {
            'onTrainingStart': function (p) {
                console.log(p.config)
            }
        }
    })
})