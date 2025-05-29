import { Layer } from './Layer.js';

export class Network {
    constructor() {
        this.layers = [];
        this.learningRate = 0.1;
        this.trainingHistory = [];
    }
    
    addLayer(neuronCount, activationFunction, inputSize = null) {
        let layerInputSize;
        if (this.layers.length === 0) {
            layerInputSize = inputSize;
        } else {
            layerInputSize = this.layers[this.layers.length - 1].size;
        }
        const layer = new Layer(neuronCount, layerInputSize, activationFunction);
        this.layers.push(layer);
        return this;
    }
    
    initializeFirstLayer(inputSize) {
        if (this.layers.length > 0 && this.layers[0].inputSize === null) {
            const firstLayer = this.layers[0];
            this.layers[0] = new Layer(
                firstLayer.size, 
                inputSize, 
                firstLayer.activationFunction
            );
        }
    }
    
    predict(inputs) {
        let currentInputs = inputs;
        for (let layer of this.layers) {
            currentInputs = layer.forward(currentInputs);
        }
        return currentInputs;
    }
    
    train(trainingData, epochs = 2000) {
        if (trainingData.length > 0) {
            this.initializeFirstLayer(trainingData[0].input.length);
        }
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalError = 0;
            for (let example of trainingData) {
                const output = this.predict(example.input);
                const error = this.calculateError(output, example.target);
                totalError += error;
                this.backpropagate(example.target);
            }
            const avgError = totalError / trainingData.length;
            this.trainingHistory.push(avgError);
        }
        return this;
    }
    
    backpropagate(targets) {
        let deltas = [];
        const outputLayer = this.layers[this.layers.length - 1];
        deltas = outputLayer.backwardOutput(targets, this.learningRate);
        for (let i = this.layers.length - 2; i >= 0; i--) {
            const currentLayer = this.layers[i];
            const nextLayer = this.layers[i + 1];
            const nextLayerWeights = nextLayer.getConnectionWeights();
            deltas = currentLayer.backwardHidden(deltas, nextLayerWeights, this.learningRate);
        }
    }
    
    calculateError(predicted, actual) {
        let sum = 0;
        for (let i = 0; i < predicted.length; i++) {
            sum += Math.pow(predicted[i] - actual[i], 2);
        }
        return sum / predicted.length;
    }
    
    setLearningRate(rate) {
        this.learningRate = rate;
        return this;
    }
    
    getInfo() {
        return {
            layers: this.layers.length,
            architecture: this.layers.map(layer => layer.size),
            learningRate: this.learningRate,
            trainedEpochs: this.trainingHistory.length
        };
    }
}
