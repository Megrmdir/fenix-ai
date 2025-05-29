import { Neuron } from './Neuron.js';

export class Layer {
    constructor(neuronCount, inputSize, activationFunction) {
        this.neurons = Array(neuronCount).fill().map(() => new Neuron(inputSize));
        this.activationFunction = activationFunction;
        this.outputs = [];
        this.size = neuronCount;
        this.inputSize = inputSize;
    }
    
    forward(inputs) {
        this.outputs = this.neurons.map(neuron => 
            neuron.forward(inputs, this.activationFunction.func)
        );
        return this.outputs;
    }
    
    backwardOutput(targets, learningRate) {
        const deltas = [];
        for (let i = 0; i < this.neurons.length; i++) {
            const error = targets[i] - this.outputs[i];
            const delta = error * this.activationFunction.derivative(this.outputs[i]);
            deltas.push(delta);
            this.neurons[i].updateWeights(delta, learningRate);
        }
        return deltas;
    }
    
    backwardHidden(nextLayerDeltas, nextLayerWeights, learningRate) {
        const deltas = [];
        for (let i = 0; i < this.neurons.length; i++) {
            let error = 0;
            for (let j = 0; j < nextLayerDeltas.length; j++) {
                error += nextLayerDeltas[j] * nextLayerWeights[j][i];
            }
            const delta = error * this.activationFunction.derivative(this.outputs[i]);
            deltas.push(delta);
            this.neurons[i].updateWeights(delta, learningRate);
        }
        return deltas;
    }
    
    getWeights() {
        return this.neurons.map(neuron => neuron.getWeights());
    }
    
    getConnectionWeights() {
        return this.neurons.map(neuron => neuron.weights);
    }
}
