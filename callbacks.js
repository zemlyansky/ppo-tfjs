class BaseCallback {
    constructor() {
        this.nCalls = 0
    }

    _onStep(alg) { return true }
    onStep(alg) {
        this.nCalls += 1
        return this._onStep(alg)
    }

    _onTrainingStart(alg) {}
    onTrainingStart(alg) {
        this._onTrainingStart(alg)
    }

    _onTrainingEnd(alg) {}
    onTrainingEnd(alg) {
        this._onTrainingEnd(alg)
    }

    _onRolloutStart(alg) {}
    onRolloutStart(alg) {
        this._onRolloutStart(alg)
    }

    _onRolloutEnd(alg) {}
    onRolloutEnd(alg) {
        this._onRolloutEnd(alg)
    }
}

class FunctionalCallback extends BaseCallback {
    constructor(callback) {
        super()
        this.callback = callback
    }

    _onStep(alg) {
        if (this.callback) {
            return this.callback(alg)
        }
        return true
    }
}

class DictCallback extends BaseCallback {
    constructor(callback) {
        super()
        this.callback = callback
    }

    _onStep(alg) {
        if (this.callback && this.callback.onStep) {
            return this.callback.onStep(alg)
        }
        return true
    }
    
    _onTrainingStart(alg) {
        if (this.callback && this.callback.onTrainingStart) {
            this.callback.onTrainingStart(alg)
        }
    }

    _onTrainingEnd(alg) {
        if (this.callback && this.callback.onTrainingEnd) {
            this.callback.onTrainingEnd(alg)
        }
    }
}

module.exports = {
    BaseCallback,
    FunctionalCallback,
    DictCallback
}