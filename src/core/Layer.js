import { Neuron } from './Neuron.js';

/**
 * Neural network layer class
 * Manages a group of neurons and their interactions
 */
export class Layer {
    /**
     * Creates a new layer
     * @param {number} neuronCount - number of neurons in layer
     * @param {number} inputSize - input vector size
     * @param {Object} activationFunction - object with activation function and its derivative
     */
    constructor(neuronCount, inputSize, activationFunction) {
        this.validateParameters(neuronCount, inputSize, activationFunction);
        
        this.neurons = this.createNeurons(neuronCount, inputSize);
        this.activationFunction = activationFunction;
        this.outputs = [];
        this.size = neuronCount;
        this.inputSize = inputSize;
    }

    /**
     * Validates constructor parameters
     * @param {number} neuronCount - number of neurons
     * @param {number} inputSize - input size
     * @param {Object} activationFunction - activation function
     */
    validateParameters(neuronCount, inputSize, activationFunction) {
        if (!Number.isInteger(neuronCount) || neuronCount <= 0) {
            throw new Error('Number of neurons must be a positive integer');
        }
        
        if (!Number.isInteger(inputSize) || inputSize <= 0) {
            throw new Error('Input size must be a positive integer');
        }
        
        if (!activationFunction || typeof activationFunction.func !== 'function' || 
            typeof activationFunction.derivative !== 'function') {
            throw new Error('Activation function must contain func and derivative methods');
        }
    }

    /**
     * Creates array of neurons
     * @param {number} neuronCount - number of neurons
     * @param {number} inputSize - input size
     * @returns {Neuron[]} array of neurons
     */
    createNeurons(neuronCount, inputSize) {
        return Array(neuronCount)
            .fill(null)
            .map(() => new Neuron(inputSize));
    }

    /**
     * Forward propagation through layer
     * @param {number[]} inputs - input values
     * @returns {number[]} layer output values
     */
    forward(inputs) {
        this.validateInputs(inputs);
        
        this.outputs = this.neurons.map(neuron => 
            neuron.forward(inputs, this.activationFunction.func)
        );
        
        return [...this.outputs];
    }

    /**
     * Backpropagation for output layer
     * @param {number[]} targets - target values
     * @param {number} learningRate - learning rate
     * @returns {number[]} error deltas
     */
    backwardOutput(targets, learningRate) {
        this.validateTargets(targets);
        
        const deltas = [];
        
        for (let i = 0; i < this.neurons.length; i++) {
            const error = targets[i] - this.outputs[i];
            const delta = error * this.activationFunction.derivative(this.outputs[i]);
            
            deltas.push(delta);
            this.neurons[i].updateWeights(delta, learningRate);
        }
        
        return deltas;
    }

    /**
     * Backpropagation for hidden layer
     * @param {number[]} nextLayerDeltas - deltas from next layer
     * @param {number[][]} nextLayerWeights - weights from next layer
     * @param {number} learningRate - learning rate
     * @returns {number[]} error deltas for current layer
     */
    backwardHidden(nextLayerDeltas, nextLayerWeights, learningRate) {
        this.validateBackpropagationInputs(nextLayerDeltas, nextLayerWeights);
        
        const deltas = [];
        
        for (let i = 0; i < this.neurons.length; i++) {
            const error = this.calculateNeuronError(i, nextLayerDeltas, nextLayerWeights);
            const delta = error * this.activationFunction.derivative(this.outputs[i]);
            
            deltas.push(delta);
            this.neurons[i].updateWeights(delta, learningRate);
        }
        
        return deltas;
    }

    /**
     * Calculates error for specific neuron
     * @param {number} neuronIndex - neuron index
     * @param {number[]} nextLayerDeltas - deltas from next layer
     * @param {number[][]} nextLayerWeights - weights from next layer
     * @returns {number} neuron error
     */
    calculateNeuronError(neuronIndex, nextLayerDeltas, nextLayerWeights) {
        return nextLayerDeltas.reduce((error, delta, deltaIndex) => {
            return error + delta * nextLayerWeights[deltaIndex][neuronIndex];
        }, 0);
    }

    /**
     * Validates input data
     * @param {number[]} inputs - input values
     */
    validateInputs(inputs) {
        if (!Array.isArray(inputs) || inputs.length !== this.inputSize) {
            throw new Error(
                `Input size (${inputs?.length}) does not match ` +
                `expected size (${this.inputSize})`
            );
        }
    }

    /**
     * Validates target values
     * @param {number[]} targets - target values
     */
    validateTargets(targets) {
        if (!Array.isArray(targets) || targets.length !== this.neurons.length) {
            throw new Error(
                `Target size (${targets?.length}) does not match ` +
                `number of neurons (${this.neurons.length})`
            );
        }
    }

    /**
     * Validates backpropagation input data
     * @param {number[]} deltas - error deltas
     * @param {number[][]} weights - weights
     */
    validateBackpropagationInputs(deltas, weights) {
        if (!Array.isArray(deltas) || !Array.isArray(weights)) {
            throw new Error('Deltas and weights must be arrays');
        }
        
        if (deltas.length !== weights.length) {
            throw new Error('Number of deltas must match number of weight vectors');
        }
    }

    /**
     * Returns weights of all neurons
     * @returns {Object[]} array of objects with weights and biases
     */
    getWeights() {
        return this.neurons.map(neuron => neuron.getWeights());
    }

    /**
     * Returns only connection weights
     * @returns {number[][]} 2D array of weights
     */
    getConnectionWeights() {
        return this.neurons.map(neuron => neuron.weights);
    }

    /**
     * Sets weights for all neurons
     * @param {Object[]} weightsData - weights data
     */
    setWeights(weightsData) {
        if (!Array.isArray(weightsData) || weightsData.length !== this.neurons.length) {
            throw new Error('Invalid weights data format');
        }
        
        weightsData.forEach((data, index) => {
            this.neurons[index].setWeights(data.weights, data.bias);
        });
    }

    /**
     * Returns layer information
     * @returns {Object} layer information
     */
    getInfo() {
        return {
            neuronCount: this.size,
            inputSize: this.inputSize,
            activationFunction: this.activationFunction.name || 'unknown',
            totalWeights: this.neurons.length * this.inputSize,
            lastOutputs: [...this.outputs]
        };
    }

    /**
     * Resets layer state
     */
    reset() {
        this.outputs = [];
        this.neurons.forEach(neuron => {
            neuron.lastInputs = null;
            neuron.lastOutput = null;
        });
    }
}
