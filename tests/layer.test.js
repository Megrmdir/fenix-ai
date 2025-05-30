import { test, expect } from "bun:test";
import { Layer } from "../src/core/Layer.js";

/**
 * Test suite for Layer class
 * Validates layer functionality including forward propagation, backpropagation, and weight management
 */

const activations = {
    sigmoid: {
        func: (x) => 1 / (1 + Math.exp(-x)),
        derivative: (y) => y * (1 - y),
        name: 'sigmoid'
    },
    relu: {
        func: (x) => Math.max(0, x),
        derivative: (y) => y > 0 ? 1 : 0,
        name: 'relu'
    },
    tanh: {
        func: (x) => Math.tanh(x),
        derivative: (y) => 1 - y * y,
        name: 'tanh'
    }
};

test("layer initializes with correct parameters", () => {
    const layer = new Layer(3, 2, activations.sigmoid);
    
    expect(layer.neurons).toHaveLength(3);
    expect(layer.size).toBe(3);
    expect(layer.inputSize).toBe(2);
    expect(layer.activationFunction).toBe(activations.sigmoid);
    expect(layer.outputs).toEqual([]);
    
    // Check that all neurons have correct input size
    layer.neurons.forEach(neuron => {
        expect(neuron.weights).toHaveLength(2);
    });
});

test("layer validates constructor parameters", () => {
    // Invalid neuron count
    expect(() => new Layer(0, 2, activations.sigmoid)).toThrow('Number of neurons must be a positive integer');
    expect(() => new Layer(-1, 2, activations.sigmoid)).toThrow('Number of neurons must be a positive integer');
    expect(() => new Layer(1.5, 2, activations.sigmoid)).toThrow('Number of neurons must be a positive integer');
    
    // Invalid input size
    expect(() => new Layer(2, 0, activations.sigmoid)).toThrow('Input size must be a positive integer');
    expect(() => new Layer(2, -1, activations.sigmoid)).toThrow('Input size must be a positive integer');
    
    // Invalid activation function
    expect(() => new Layer(2, 2, {})).toThrow('Activation function must contain func and derivative methods');
    expect(() => new Layer(2, 2, { func: () => {} })).toThrow('Activation function must contain func and derivative methods');
});

test("layer performs forward propagation correctly", () => {
    const layer = new Layer(2, 2, activations.sigmoid);
    
    // Set known weights for predictable output
    layer.neurons[0].setWeights([0.5, 0.3], 0.1);
    layer.neurons[1].setWeights([-0.2, 0.4], -0.1);
    
    const inputs = [1.0, 0.5];
    const outputs = layer.forward(inputs);
    
    expect(outputs).toHaveLength(2);
    expect(layer.outputs).toEqual(outputs);
    
    // Verify calculations
    const sigmoid = activations.sigmoid.func;
    const expectedOutput1 = sigmoid(1.0 * 0.5 + 0.5 * 0.3 + 0.1); // sigmoid(0.75)
    const expectedOutput2 = sigmoid(1.0 * (-0.2) + 0.5 * 0.4 + (-0.1)); // sigmoid(-0.1)
    
    expect(outputs[0]).toBeCloseTo(expectedOutput1, 6);
    expect(outputs[1]).toBeCloseTo(expectedOutput2, 6);
});

test("layer validates input size during forward propagation", () => {
    const layer = new Layer(2, 3, activations.sigmoid);
    
    // Valid input
    expect(() => layer.forward([1, 2, 3])).not.toThrow();
    
    // Invalid input sizes
    expect(() => layer.forward([1, 2])).toThrow('Input size (2) does not match expected size (3)');
    expect(() => layer.forward([1, 2, 3, 4])).toThrow('Input size (4) does not match expected size (3)');
    
    // Invalid input type (string "invalid" has length 7)
    expect(() => layer.forward("invalid")).toThrow('Input size (7) does not match expected size (3)');
});

test("layer performs output layer backpropagation correctly", () => {
    const layer = new Layer(2, 2, activations.sigmoid);
    
    // Set initial weights
    layer.neurons[0].setWeights([0.5, 0.3], 0.1);
    layer.neurons[1].setWeights([-0.2, 0.4], -0.1);
    
    // Forward pass
    const inputs = [1.0, 0.5];
    const outputs = layer.forward(inputs);
    
    // Backpropagation
    const targets = [0.8, 0.2];
    const learningRate = 0.1;
    const deltas = layer.backwardOutput(targets, learningRate);
    
    expect(deltas).toHaveLength(2);
    
    // Verify delta calculations
    const derivative = activations.sigmoid.derivative;
    const expectedDelta1 = (targets[0] - outputs[0]) * derivative(outputs[0]);
    const expectedDelta2 = (targets[1] - outputs[1]) * derivative(outputs[1]);
    
    expect(deltas[0]).toBeCloseTo(expectedDelta1, 6);
    expect(deltas[1]).toBeCloseTo(expectedDelta2, 6);
});

test("layer performs hidden layer backpropagation correctly", () => {
    const layer = new Layer(2, 2, activations.sigmoid);
    
    // Forward pass
    const inputs = [1.0, 0.5];
    const outputs = layer.forward(inputs);
    
    // Simulate next layer data
    const nextLayerDeltas = [0.1, -0.05, 0.02];
    const nextLayerWeights = [
        [0.3, -0.2],  // weights from neuron 0 of next layer
        [0.1, 0.4],   // weights from neuron 1 of next layer
        [-0.1, 0.2]   // weights from neuron 2 of next layer
    ];
    
    const learningRate = 0.1;
    const deltas = layer.backwardHidden(nextLayerDeltas, nextLayerWeights, learningRate);
    
    expect(deltas).toHaveLength(2);
    
    // Verify error calculation for first neuron
    const expectedError1 = nextLayerDeltas[0] * nextLayerWeights[0][0] + 
                          nextLayerDeltas[1] * nextLayerWeights[1][0] + 
                          nextLayerDeltas[2] * nextLayerWeights[2][0];
    const expectedDelta1 = expectedError1 * activations.sigmoid.derivative(outputs[0]);
    
    expect(deltas[0]).toBeCloseTo(expectedDelta1, 6);
});

test("layer validates backpropagation inputs", () => {
    const layer = new Layer(2, 2, activations.sigmoid);
    layer.forward([1, 1]); // Need forward pass first
    
    // Invalid target size for output backprop
    expect(() => layer.backwardOutput([0.5], 0.1)).toThrow('Target size (1) does not match number of neurons (2)');
    expect(() => layer.backwardOutput([0.5, 0.3, 0.1], 0.1)).toThrow('Target size (3) does not match number of neurons (2)');
    
    // Invalid inputs for hidden backprop
    expect(() => layer.backwardHidden("invalid", [[1, 2]], 0.1)).toThrow('Deltas and weights must be arrays');
    expect(() => layer.backwardHidden([0.1], "invalid", 0.1)).toThrow('Deltas and weights must be arrays');
    expect(() => layer.backwardHidden([0.1, 0.2], [[1, 2]], 0.1)).toThrow('Number of deltas must match number of weight vectors');
});

test("layer weight management works correctly", () => {
    const layer = new Layer(2, 2, activations.sigmoid);
    
    const newWeights = [
        { weights: [0.1, 0.2], bias: 0.3 },
        { weights: [0.4, 0.5], bias: 0.6 }
    ];
    
    layer.setWeights(newWeights);
    const retrievedWeights = layer.getWeights();
    
    expect(retrievedWeights).toHaveLength(2);
    expect(retrievedWeights[0].weights).toEqual([0.1, 0.2]);
    expect(retrievedWeights[0].bias).toBe(0.3);
    expect(retrievedWeights[1].weights).toEqual([0.4, 0.5]);
    expect(retrievedWeights[1].bias).toBe(0.6);
});

test("layer provides connection weights correctly", () => {
    const layer = new Layer(2, 3, activations.relu);
    
    const connectionWeights = layer.getConnectionWeights();
    
    expect(connectionWeights).toHaveLength(2);
    expect(connectionWeights[0]).toHaveLength(3);
    expect(connectionWeights[1]).toHaveLength(3);
    
    // Should be references to actual neuron weights
    connectionWeights[0][0] = 999;
    expect(layer.neurons[0].weights[0]).toBe(999);
});

test("layer provides correct information", () => {
    const layer = new Layer(3, 2, activations.tanh);
    
    const info = layer.getInfo();
    
    expect(info.neuronCount).toBe(3);
    expect(info.inputSize).toBe(2);
    expect(info.activationFunction).toBe('tanh');
    expect(info.totalWeights).toBe(6); // 3 neurons * 2 inputs each
    expect(info.lastOutputs).toEqual([]);
    
    // After forward pass
    layer.forward([1, -1]);
    const updatedInfo = layer.getInfo();
    expect(updatedInfo.lastOutputs).toHaveLength(3);
});

test("layer reset functionality works correctly", () => {
    const layer = new Layer(2, 2, activations.sigmoid);
    
    // Perform forward pass
    layer.forward([1, 1]);
    expect(layer.outputs).not.toEqual([]);
    expect(layer.neurons[0].lastInputs).not.toBeNull();
    expect(layer.neurons[0].lastOutput).not.toBeNull();
    
    // Reset
    layer.reset();
    expect(layer.outputs).toEqual([]);
    expect(layer.neurons[0].lastInputs).toBeNull();
    expect(layer.neurons[0].lastOutput).toBeNull();
});

test("layer handles different activation functions", () => {
    const reluLayer = new Layer(1, 1, activations.relu);
    const tanhLayer = new Layer(1, 1, activations.tanh);
    
    // Set weights for predictable results
    reluLayer.neurons[0].setWeights([1], -0.5);
    tanhLayer.neurons[0].setWeights([1], -0.5);
    
    const input = [1];
    
    const reluOutput = reluLayer.forward(input);
    const tanhOutput = tanhLayer.forward(input);
    
    // ReLU: max(0, 1*1 + (-0.5)) = max(0, 0.5) = 0.5
    expect(reluOutput[0]).toBeCloseTo(0.5, 6);
    
    // Tanh: tanh(1*1 + (-0.5)) = tanh(0.5)
    expect(tanhOutput[0]).toBeCloseTo(Math.tanh(0.5), 6);
});
