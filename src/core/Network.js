import { Layer } from './Layer.js';

/**
 * Neural network class
 * Manages network architecture, training and predictions
 */
export class Network {
    /**
     * Creates a new neural network
     */
    constructor() {
        this.layers = [];
        this.learningRate = 0.1;
        this.trainingHistory = [];
        this.isCompiled = false;
    }

    /**
     * Adds a new layer to the network
     * @param {number} neuronCount - number of neurons in layer
     * @param {Object} activationFunction - activation function
     * @param {number|null} inputSize - input size (only for first layer)
     * @returns {Network} returns this for method chaining
     */
    addLayer(neuronCount, activationFunction, inputSize = null) {
        this.validateLayerParameters(neuronCount, activationFunction, inputSize);
        
        const layerInputSize = this.determineLayerInputSize(inputSize);
        const layer = new Layer(neuronCount, layerInputSize, activationFunction);
        
        this.layers.push(layer);
        this.isCompiled = false;
        
        return this;
    }

    /**
     * Validates layer parameters
     * @param {number} neuronCount - number of neurons
     * @param {Object} activationFunction - activation function
     * @param {number|null} inputSize - input size
     */
    validateLayerParameters(neuronCount, activationFunction, inputSize) {
        if (!Number.isInteger(neuronCount) || neuronCount <= 0) {
            throw new Error('Number of neurons must be a positive integer');
        }
        
        if (!activationFunction || typeof activationFunction.func !== 'function') {
            throw new Error('Activation function must be an object with func method');
        }
        
        if (this.layers.length === 0 && (inputSize === null || inputSize <= 0)) {
            throw new Error('Input size must be specified for the first layer');
        }
    }

    /**
     * Determines input size for new layer
     * @param {number|null} inputSize - explicitly specified input size
     * @returns {number} input size for layer
     */
    determineLayerInputSize(inputSize) {
        if (this.layers.length === 0) {
            return inputSize;
        }
        
        return this.layers[this.layers.length - 1].size;
    }

    /**
     * Compiles the network for training
     */
    compile() {
        if (this.layers.length === 0) {
            throw new Error('Network must contain at least one layer');
        }
        
        this.isCompiled = true;
    }

    /**
     * Makes predictions for input data
     * @param {number[]} inputs - input data
     * @returns {number[]} prediction result
     */
    predict(inputs) {
        this.validatePredictionInputs(inputs);
        
        let currentInputs = [...inputs];
        
        for (const layer of this.layers) {
            currentInputs = layer.forward(currentInputs);
        }
        
        return currentInputs;
    }

    /**
     * Trains the network on provided data
     * @param {Object[]} trainingData - training data
     * @param {number} epochs - number of training epochs
     * @param {Object} options - additional training parameters
     * @returns {Network} returns this for method chaining
     */
    train(trainingData, epochs = 2000, options = {}) {
        this.validateTrainingData(trainingData);
        this.prepareForTraining(trainingData);
        
        const {
            verbose = false,
            validationData = null,
            earlyStoppingPatience = null
        } = options;
        
        let bestValidationError = Infinity;
        let patienceCounter = 0;
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            const epochError = this.trainEpoch(trainingData);
            this.trainingHistory.push(epochError);
            
            // Validation and early stopping
            if (validationData && earlyStoppingPatience) {
                const validationError = this.evaluateValidation(validationData);
                
                if (validationError < bestValidationError) {
                    bestValidationError = validationError;
                    patienceCounter = 0;
                } else {
                    patienceCounter++;
                    
                    if (patienceCounter >= earlyStoppingPatience) {
                        if (verbose) {
                            console.log(`Early stopping at epoch ${epoch + 1}`);
                        }
                        break;
                    }
                }
            }
            
            // Progress output
            if (verbose && (epoch + 1) % Math.max(1, Math.floor(epochs / 10)) === 0) {
                console.log(`Epoch ${epoch + 1}/${epochs}, Error: ${epochError.toFixed(6)}`);
            }
        }
        
        return this;
    }

    /**
     * Trains the network for one epoch
     * @param {Object[]} trainingData - training data
     * @returns {number} average error for epoch
     */
    trainEpoch(trainingData) {
        let totalError = 0;
        
        // Shuffle data for better training
        const shuffledData = this.shuffleArray([...trainingData]);
        
        for (const example of shuffledData) {
            const output = this.predict(example.input);
            const error = this.calculateError(output, example.target);
            
            totalError += error;
            this.backpropagate(example.target);
        }
        
        return totalError / trainingData.length;
    }

    /**
     * Shuffles array using Fisher-Yates algorithm
     * @param {Array} array - array to shuffle
     * @returns {Array} shuffled array
     */
    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }

    /**
     * Performs backpropagation
     * @param {number[]} targets - target values
     */
    backpropagate(targets) {
        let deltas = [];
        
        // Backpropagation for output layer
        const outputLayer = this.layers[this.layers.length - 1];
        deltas = outputLayer.backwardOutput(targets, this.learningRate);
        
        // Backpropagation for hidden layers
        for (let i = this.layers.length - 2; i >= 0; i--) {
            const currentLayer = this.layers[i];
            const nextLayer = this.layers[i + 1];
            const nextLayerWeights = nextLayer.getConnectionWeights();
            
            deltas = currentLayer.backwardHidden(deltas, nextLayerWeights, this.learningRate);
        }
    }

    /**
     * Calculates mean squared error
     * @param {number[]} predicted - predicted values
     * @param {number[]} actual - actual values
     * @returns {number} error value
     */
    calculateError(predicted, actual) {
        if (predicted.length !== actual.length) {
            throw new Error('Predicted and actual values must have the same length');
        }
        
        const sumSquaredError = predicted.reduce((sum, pred, index) => {
            return sum + Math.pow(pred - actual[index], 2);
        }, 0);
        
        return sumSquaredError / predicted.length;
    }

    /**
     * Evaluates performance on validation data
     * @param {Object[]} validationData - validation data
     * @returns {number} average validation error
     */
    evaluateValidation(validationData) {
        let totalError = 0;
        
        for (const example of validationData) {
            const output = this.predict(example.input);
            const error = this.calculateError(output, example.target);
            totalError += error;
        }
        
        return totalError / validationData.length;
    }

    /**
     * Validates prediction input data
     * @param {number[]} inputs - input data
     */
    validatePredictionInputs(inputs) {
        if (!Array.isArray(inputs)) {
            throw new Error('Input data must be an array');
        }
        
        if (this.layers.length === 0) {
            throw new Error('Network contains no layers');
        }
        
        const expectedInputSize = this.layers[0].inputSize;
        if (inputs.length !== expectedInputSize) {
            throw new Error(
                `Input size (${inputs.length}) does not match ` +
                `expected size (${expectedInputSize})`
            );
        }
    }

    /**
     * Validates training data
     * @param {Object[]} trainingData - training data
     */
    validateTrainingData(trainingData) {
        if (!Array.isArray(trainingData) || trainingData.length === 0) {
            throw new Error('Training data must be a non-empty array');
        }
        
        const firstExample = trainingData[0];
        if (!firstExample.input || !firstExample.target) {
            throw new Error('Each example must contain input and target fields');
        }
        
        // Check size consistency
        const inputSize = firstExample.input.length;
        const outputSize = firstExample.target.length;
        
        for (const example of trainingData) {
            if (example.input.length !== inputSize) {
                throw new Error('All input vectors must have the same size');
            }
            
            if (example.target.length !== outputSize) {
                throw new Error('All target vectors must have the same size');
            }
        }
    }

    /**
     * Prepares network for training
     * @param {Object[]} trainingData - training data
     */
    prepareForTraining(trainingData) {
        if (!this.isCompiled) {
            this.compile();
        }
        
        this.trainingHistory = [];
        
        // Check architecture compatibility with data
        const inputSize = trainingData[0].input.length;
        const outputSize = trainingData[0].target.length;
        
        if (this.layers[0].inputSize !== inputSize) {
            throw new Error(
                `Network input size (${this.layers[0].inputSize}) does not match ` +
                `input data size (${inputSize})`
            );
        }
        
        const lastLayerSize = this.layers[this.layers.length - 1].size;
        if (lastLayerSize !== outputSize) {
            throw new Error(
                `Output layer size (${lastLayerSize}) does not match ` +
                `target data size (${outputSize})`
            );
        }
    }

    /**
     * Sets learning rate
     * @param {number} rate - new learning rate
     * @returns {Network} returns this for method chaining
     */
    setLearningRate(rate) {
        if (typeof rate !== 'number' || rate <= 0) {
            throw new Error('Learning rate must be a positive number');
        }
        
        this.learningRate = rate;
        return this;
    }

    /**
     * Returns detailed network information
     * @returns {Object} network information
     */
    getInfo() {
        const totalParameters = this.layers.reduce((total, layer) => {
            return total + (layer.size * layer.inputSize) + layer.size; // weights + biases
        }, 0);
        
        return {
            layers: this.layers.length,
            architecture: this.layers.map(layer => layer.size),
            inputSize: this.layers.length > 0 ? this.layers[0].inputSize : 0,
            outputSize: this.layers.length > 0 ? this.layers[this.layers.length - 1].size : 0,
            totalParameters,
            learningRate: this.learningRate,
            isCompiled: this.isCompiled,
            trainedEpochs: this.trainingHistory.length,
            lastError: this.trainingHistory.length > 0 ? 
                this.trainingHistory[this.trainingHistory.length - 1] : null
        };
    }

    /**
     * Exports model to JSON
     * @returns {string} JSON representation of model
     */
    exportModel() {
        const modelData = {
            architecture: this.layers.map(layer => ({
                neuronCount: layer.size,
                inputSize: layer.inputSize,
                activationFunction: layer.activationFunction.name || 'unknown',
                weights: layer.getWeights()
            })),
            learningRate: this.learningRate,
            trainingHistory: this.trainingHistory
        };
        
        return JSON.stringify(modelData, null, 2);
    }

    /**
     * Resets network state
     */
    reset() {
        this.layers.forEach(layer => layer.reset());
        this.trainingHistory = [];
    }

    /**
     * Returns training history
     * @returns {number[]} array of error values by epochs
     */
    getTrainingHistory() {
        return [...this.trainingHistory];
    }
}
