/**
 * Neuron class - basic element of neural network
 * Implements forward propagation and weight updates
 */
export class Neuron {
    /**
     * Creates a new neuron
     * @param {number} inputSize - number of input connections
     */
    constructor(inputSize) {
        this.weights = this.initializeWeights(inputSize);
        this.bias = this.initializeBias();
        this.lastInputs = null;
        this.lastOutput = null;
    }

    /**
     * Initializes weights using Xavier/Glorot method
     * @param {number} inputSize - size of input vector
     * @returns {number[]} array of weights
     */
    initializeWeights(inputSize) {
        const limit = Math.sqrt(6 / (inputSize + 1));
        return Array(inputSize)
            .fill(0)
            .map(() => (Math.random() * 2 - 1) * limit);
    }

    /**
     * Initializes bias with small random value
     * @returns {number} bias value
     */
    initializeBias() {
        return (Math.random() * 2 - 1) * 0.1;
    }

    /**
     * Forward propagation of signal
     * @param {number[]} inputs - input values
     * @param {Function} activationFunction - activation function
     * @returns {number} neuron output value
     */
    forward(inputs, activationFunction) {
        this.validateInputs(inputs);
        
        this.lastInputs = [...inputs];
        
        const weightedSum = this.calculateWeightedSum(inputs);
        this.lastOutput = activationFunction(weightedSum);
        
        return this.lastOutput;
    }

    /**
     * Calculates weighted sum of inputs
     * @param {number[]} inputs - input values
     * @returns {number} weighted sum + bias
     */
    calculateWeightedSum(inputs) {
        return inputs.reduce((sum, input, index) => {
            return sum + input * this.weights[index];
        }, this.bias);
    }

    /**
     * Updates weights and bias based on error delta
     * @param {number} delta - error delta
     * @param {number} learningRate - learning rate
     */
    updateWeights(delta, learningRate) {
        if (!this.lastInputs) {
            throw new Error('Cannot update weights: no last inputs data available');
        }

        // Update weights
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] += learningRate * delta * this.lastInputs[i];
        }

        // Update bias
        this.bias += learningRate * delta;
    }

    /**
     * Validates input data correctness
     * @param {number[]} inputs - input values
     */
    validateInputs(inputs) {
        if (!Array.isArray(inputs)) {
            throw new Error('Input data must be an array');
        }
        
        if (inputs.length !== this.weights.length) {
            throw new Error(
                `Input size (${inputs.length}) does not match ` +
                `number of weights (${this.weights.length})`
            );
        }

        if (inputs.some(input => typeof input !== 'number' || !isFinite(input))) {
            throw new Error('All input values must be finite numbers');
        }
    }

    /**
     * Returns current weights and bias
     * @returns {Object} object with weights and bias
     */
    getWeights() {
        return {
            weights: [...this.weights],
            bias: this.bias
        };
    }

    /**
     * Sets new weights and bias
     * @param {number[]} weights - new weights
     * @param {number} bias - new bias
     */
    setWeights(weights, bias) {
        if (!Array.isArray(weights) || weights.length !== this.weights.length) {
            throw new Error('Invalid weights format');
        }
        
        this.weights = [...weights];
        this.bias = bias;
    }

    /**
     * Returns neuron information
     * @returns {Object} neuron information
     */
    getInfo() {
        return {
            inputSize: this.weights.length,
            weightsSum: this.weights.reduce((sum, w) => sum + Math.abs(w), 0),
            bias: this.bias,
            lastOutput: this.lastOutput
        };
    }
}
