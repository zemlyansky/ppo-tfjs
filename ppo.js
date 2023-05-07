const tf = require('@tensorflow/tfjs')
// const nqdm = require('nqdm')

const { FunctionalCallback, DictCallback, BaseCallback } = require('./callbacks.js')

function log () {
    console.log('[PPO]', ...arguments)
}

function logProbCategorical(logits, x) {
    const numActions = logits.shape[1]
    const logprobabilitiesAll = tf.logSoftmax(logits)
    var logprobability = tf.sum(
        tf.mul(tf.oneHot(x, numActions), logprobabilitiesAll), 1 //axis
    )
    return logprobability
}

function logProbNormal(loc, scale, x) {
    const logUnnormalized = tf.mul(
        -0.5,
        tf.square(
            tf.sub(
                tf.div(x, scale),
                tf.div(loc, scale)
            )
        )
    )
    const logNormalization = tf.add(
        tf.scalar(0.5 * Math.log(2.0 * Math.PI)),
        tf.log(scale)
    )
    return tf.sub(logUnnormalized, logNormalization)
}



function discountedCumulativeSums (arr, coeff) {
    var res = []
    var s = 0
    arr.reverse().forEach(v => {
        s = v + s * coeff
        res.push(s)
    })
    return res.reverse()
}

const bufferConfigDefault = {
    gamma: 0.99,
    lam: 0.95
}

class Buffer {
    constructor(bufferConfig) {
        this.bufferConfig = Object.assign({}, bufferConfigDefault, bufferConfig)
        this.gamma = this.bufferConfig.gamma
        this.lam = this.bufferConfig.lam
        this.reset()
    }

    add(observation, action, reward, value, logprobability) {
        // log('>', observation)
        this.observationBuffer.push(observation.slice(0))
        this.actionBuffer.push(action)
        this.rewardBuffer.push(reward)
        this.valueBuffer.push(value)
        this.logprobabilityBuffer.push(logprobability)
        this.pointer += 1
    }

    finishTrajectory(lastValue) {
        log('>', this.rewardBuffer, this.rewardBuffer.length)
        var rewards = this.rewardBuffer
            .slice(this.trajectoryStartIndex, this.pointer)
            .concat(lastValue * this.gamma)
        log('>', rewards, rewards.length)
        var values = this.valueBuffer
            .slice(this.trajectoryStartIndex, this.pointer)
            .concat(lastValue)
        log('>', values, values.length)
        var deltas = rewards
            .slice(0, -1)
            .map((reward, ri) => reward - (values[ri] - this.gamma * values[ri + 1]))
        log('deltas>', deltas.length)
        
        this.advantageBuffer = this.advantageBuffer
            .concat(discountedCumulativeSums(deltas, this.gamma * this.lam))
        this.returnBuffer = this.returnBuffer
            .concat(discountedCumulativeSums(rewards, this.gamma).slice(0, -1))
                
        this.trajectoryStartIndex = this.pointer

        // log('>', this.observationBuffer.length)
        // log('>', this.actionBuffer.length)
        // log('>', this.advantageBuffer.length)
        // log('>', this.returnBuffer.length)
        // log('>', this.rewardBuffer.length)
        // log('>', deltas.length)

        // process.exit(0)
    }

    get() {
        var advantageMean = tf.mean(this.advantageBuffer).arraySync()
        var advantageStd = tf.moments(this.advantageBuffer).variance.sqrt().arraySync()
        
        this.advantageBuffer = this.advantageBuffer
            .map(advantage => (advantage - advantageMean) / advantageStd)
        
        return [
            this.observationBuffer,
            this.actionBuffer,
            this.advantageBuffer,
            this.returnBuffer,
            this.logprobabilityBuffer
        ]
    }

    reset() {
        this.observationBuffer = []
        this.actionBuffer = []
        this.advantageBuffer = []
        this.rewardBuffer = []
        this.returnBuffer = []
        this.valueBuffer = []
        this.logprobabilityBuffer = []
        this.trajectoryStartIndex = 0
        this.pointer = 0
    }

}

class ActorCriticPolicy {
    constructor(policyConfig) {
        const policyConfigDefault = {
            netArch: {
                'pi': [16, 16],
                'vf': [16, 16]
            },
            activationFn: 'tanh',
            shareFeaturesExtractor: true,
        }
        this.policyConfig = Object.assign({}, policyConfigDefault, policyConfig)
        this.observationSpace = this.policyConfig.observationSpace
        this.actionSpace = this.policyConfig.actionSpace
        this.netArch = this.policyConfig.netArch
        this.activationFn = this.policyConfig.activationFn
        this.shareFeaturesExtractor = this.policyConfig.shareFeaturesExtractor
    }
    
    _makeFeaturesExtractor() {
    }
}

class PPO {
    constructor(env, config) {
        const configDefault = {
            nSteps: 200,
            policyIterations: 80,
            policyLearningRate: 3e-4,
            valueIterations: 80,
            valueLearningRate: 1e-3,
            clipRatio: 0.2,
            targetKL: 0.01,
            useSDE: false, // TODO: State Dependent Exploration (gSDE)
        }

        this.config = Object.assign({}, configDefault, config)
        this.env = env
        this.numTimesteps = 0
        this.lastObservation = null

        var input = tf.layers.input({shape: this.env.observationSpace.shape})
        var l = tf.layers.dense({units: 16, activation: 'relu'}).apply(input)
        if (this.env.actionSpace.class == 'Discrete') {
            l = tf.layers.dense({units: this.env.actionSpace.n, activation: 'linear'}).apply(l)
        } else if (this.env.actionSpace.class == 'Box') {
            l = tf.layers.dense({units: this.env.actionSpace.shape[0], activation: 'linear'}).apply(l)
        } else {
            throw new Error('Unknown action space class: ' + this.env.actionSpace.class)
        }
        this.actor = tf.model({inputs: input, outputs: l})

        var input = tf.layers.input({shape: this.env.observationSpace.shape})
        var l = tf.layers.dense({units: 16, activation: 'relu'}).apply(input)
        var l = tf.layers.dense({units: 1, activation: 'linear'}).apply(l)
        this.critic = tf.model({inputs: input, outputs: l})

        if (this.env.actionSpace.class == 'Box') {
            this.logStd = tf.variable(tf.zeros([1, this.env.actionSpace.shape[0]]))
        }

        this.buffer = new Buffer(config)

        this.optPolicy = tf.train.adam(this.config.policyLearningRate)
        this.optValue = tf.train.adam(this.config.valueLearningRate)
    }

    sampleAction(observation) {
        const preds = this.actor.predict(tf.tensor([observation]))
        let action 
        if (this.env.actionSpace.class == 'Discrete') {
            action = tf.multinomial(preds, 1)
        } else if (this.env.actionSpace.class == 'Box') {
            action = tf.add(
                tf.mul(
                    tf.randomNormal([1, this.env.actionSpace.shape[0]]), 
                    tf.exp(this.logStd)
                ),
                preds
            )
        }
        return [preds, action]
    }

    logProb(preds, actions) {
        // Preds can be logits or means
        if (this.env.actionSpace.class == 'Discrete') {
            return logProbCategorical(preds, actions)
        } else if (this.env.actionSpace.class == 'Box') {
            return logProbNormal(preds, tf.exp(this.logStd), actions)
        }
    }
    
    predict(observation, deterministic=false) {
        return this.actor.predict(observation)
    }

    trainPolicy(observationBuffer, actionBuffer, logprobabilityBuffer, advantageBuffer) {
        observationBuffer = tf.tensor(observationBuffer)
    
        var optFunc = () => {
            var preds = this.actor.predict(observationBuffer) // -> Logits or means
            var ratio = tf.exp(tf.sub(
                this.logProb(preds, actionBuffer),
                logprobabilityBuffer
            ))
            var minAdvantage = tf.where(
                tf.greater(advantageBuffer, 0),
                tf.mul(tf.add(1, this.config.clipRatio), advantageBuffer),
                tf.mul(tf.sub(1, this.config.clipRatio), advantageBuffer)
            )
            var policyLoss = tf.neg(tf.mean(
                tf.minimum(tf.mul(ratio, advantageBuffer), minAdvantage)
            ))
            return policyLoss
        }
    
        tf.tidy(() => {
            var {values, grads} = this.optPolicy.computeGradients(optFunc)
            this.optPolicy.applyGradients(grads)
        })
    
        var kl = tf.mean(tf.sub(
            logprobabilityBuffer,
            this.logProb(this.actor.predict(observationBuffer), actionBuffer)
        ))
        // kl = tf.sum(kl) // TODO: ?
    
        return kl
    }

    trainValue(observationBuffer, returnBuffer) {
        observationBuffer = tf.tensor(observationBuffer)
        returnBuffer = tf.tensor(returnBuffer).reshape([-1, 1])
    
        var optFunc = () => {
            const valuesPred = this.critic.predict(observationBuffer)
            const loss = tf.losses.meanSquaredError(returnBuffer, valuesPred)
            return loss
        }
                
        tf.tidy(() => {
            var {values, grads} = this.optValue.computeGradients(optFunc)
            this.optValue.applyGradients(grads)
        })
    }

    _initCallback(callback) {
        // Function, not class
        if (typeof callback === 'function') {
            if (callback.prototype.constructor === undefined) {
                return new FunctionalCallback(callback)
            }
            return callback
        }
        if (typeof callback === 'object') {
            return new DictCallback(callback)
        }
        return new BaseCallback() 
        // TODO:List
        // TODO:Class
    }

    async collectRollouts(callback) {
        if (this.lastObservation === null) {
            this.lastObservation = this.env.reset()
        }

        this.buffer.reset()
        callback.onRolloutStart(this)

        var sumReturn = 0
        var sumLength = 0
        var numEpisodes = 0

        for (let step = 0; step < this.config.nSteps; step++) {
            // Predict action, value and logprob from last observation
            var [preds, action] = this.sampleAction(this.lastObservation)
            console.log('preds:', preds.arraySync(), preds.shape)
            console.log('action:', action.arraySync(), action.shape)
            action = action.arraySync()[0][0]
            var valueT = this.critic.predict(tf.tensor([this.lastObservation]))
            console.log('valueT:', valueT.arraySync(), valueT.shape)
            valueT = valueT.arraySync()[0][0]        
            var logprobabilityT = this.logProb(preds, action)
            console.log('logprobabilityT:', logprobabilityT.arraySync(), logprobabilityT.shape)
            logprobabilityT = logprobabilityT.arraySync()[0]
            process.exit()

            // TODO: Rescale for continuous action space

            // Take action in environment
            var [newObservation, reward, done] = await this.env.step(action)
            sumReturn += reward
            sumLength += 1

            // Update global timestep counter
            this.numTimesteps += 1 

            callback.onStep(this)

            this.buffer.add(
                this.lastObservation, 
                action, 
                reward, 
                valueT, 
                logprobabilityT
            )
            
            this.lastObservation = newObservation
            
            if (done || step === this.config.nSteps - 1) {
                //log('end:', observation)
                const lastValue = done ? 0 : this.critic.predict(tf.tensor([newObservation])).arraySync()[0][0]
                this.buffer.finishTrajectory(lastValue)
                numEpisodes += 1
                this.lastObservation = this.env.reset()
                // log('Start:', observation)
            }
        }
            
        log(`Timesteps: ${this.numTimesteps}, Episodes: ${numEpisodes}`)
        log(`Avg returns: ${sumReturn / numEpisodes}`)
        log(`Avg length: ${sumLength / numEpisodes}`)

        callback.onRolloutEnd(this)
    }

    async train(config) {
        // Get values from the buffer
        var [
            observationBuffer,
            actionBuffer,
            advantageBuffer,
            returnBuffer,
            logprobabilityBuffer,
        ] = this.buffer.get()

        log('Train policy net...')
        for (let i = 0; i < this.config.policyIterations; i++) {
            var kl = this.trainPolicy(observationBuffer, actionBuffer, logprobabilityBuffer, advantageBuffer)
            if (kl > 1.5 * this.config.targetKL) {
                log('Break')
                break
            }
        }

        log('Train value net...')
        for (let i = 0;  i < this.config.valueIterations; i++) {
            this.trainValue(observationBuffer, returnBuffer)
        }
    }

    async learn(learnConfig) {
        const learnConfigDefault = {
            'totalTimesteps': 1000,
            'logInterval': 1,
            'callback': null
        }
        let { 
            totalTimesteps,
            logInterval,
            callback
        } = Object.assign({}, learnConfigDefault, learnConfig)

        callback = this._initCallback(callback)
        
        let iteration = 0
        
        callback.onTrainingStart(this)

        log('Start')

        while (this.numTimesteps < totalTimesteps) {
            await this.collectRollouts(callback)
            iteration += 1
            if (logInterval && iteration % logInterval === 0) {
                log(`Timesteps: ${this.numTimesteps}`)
            }
            this.train()
        }
        
        callback.onTrainingEnd(this)
    }
}

module.exports = PPO