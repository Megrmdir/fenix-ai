import { test, expect } from "bun:test";
import { Neuron } from "../src/core/Neuron.js";

/**
 * Test suite for Neuron class
 * Validates core neuron functionality including initialization, forward pass, and weight updates
 */

test("neuron initializes with correct weights and bias", () => {
    const neuron = new Neuron(3);
    
    expect(neuron.weights).toHaveLength(3);
    expect(typeof neuron.bias).toBe('number');
    expect(neuron.lastInputs).toBeNull();
    expect(neuron.lastOutput).toBeNull();
    
    // Check Xavier initialization bounds
    const limit = Math.sqrt(6 / (3 + 1));
    neuron.weights.forEach(weight => {
        expect(Math.abs(weight)).toBeLessThanOrEqual(limit);
    });
    
    expect(Math.abs(neuron.bias)).toBeLessThanOrEqual(0.1);
});

test("neuron performs forward propagation correctly", () => {
    const neuron = new Neuron(2);
    neuron.setWeights([0.5, -0.3], 0.1);
    
    const sigmoid = (x) => 1 / (1 + Math.exp(-x));
    const inputs = [1.0, 0.5];
    
    const output = neuron.forward(inputs, sigmoid);
    
    // Expected: sigmoid(1.0 * 0.5 + 0.5 * (-0.3) + 0.1) = sigmoid(0.45)
    const expectedOutput = sigmoid(0.45);
    
    expect(output).toBeCloseTo(expectedOutput, 6);
    expect(neuron.lastInputs).toEqual(inputs);
    expect(neuron.lastOutput).toBeCloseTo(expectedOutput, 6);
});

test("neuron validates input size correctly", () => {
    const neuron = new Neuron(2);
    const sigmoid = (x) => 1 / (1 + Math.exp(-x));
    
    // Valid input
    expect(() => neuron.forward([1, 2], sigmoid)).not.toThrow();
    
    // Invalid input sizes
    expect(() => neuron.forward([1], sigmoid)).toThrow('Input size (1) does not match number of weights (2)');
    expect(() => neuron.forward([1, 2, 3], sigmoid)).toThrow('Input size (3) does not match number of weights (2)');
    
    // Invalid input types
    expect(() => neuron.forward("invalid", sigmoid)).toThrow('Input data must be an array');
    expect(() => neuron.forward([1, "invalid"], sigmoid)).toThrow('All input values must be finite numbers');
    expect(() => neuron.forward([1, Infinity], sigmoid)).toThrow('All input values must be finite numbers');
});

test("neuron updates weights correctly during backpropagation", () => {
    const neuron = new Neuron(2);
    const initialWeights = [0.5, -0.3];
    const initialBias = 0.1;
    
    neuron.setWeights([...initialWeights], initialBias);
    
    // Perform forward pass first
    const inputs = [1.0, 0.5];
    const sigmoid = (x) => 1 / (1 + Math.exp(-x));
    neuron.forward(inputs, sigmoid);
    
    // Update weights
    const delta = 0.2;
    const learningRate = 0.1;
    neuron.updateWeights(delta, learningRate);
    
    // Check weight updates: new_weight = old_weight + learning_rate * delta * input
    const expectedWeights = [
        initialWeights[0] + learningRate * delta * inputs[0], // 0.5 + 0.1 * 0.2 * 1.0 = 0.52
        initialWeights[1] + learningRate * delta * inputs[1]  // -0.3 + 0.1 * 0.2 * 0.5 = -0.29
    ];
    const expectedBias = initialBias + learningRate * delta; // 0.1 + 0.1 * 0.2 = 0.12
    
    expect(neuron.weights[0]).toBeCloseTo(expectedWeights[0], 6);
    expect(neuron.weights[1]).toBeCloseTo(expectedWeights[1], 6);
    expect(neuron.bias).toBeCloseTo(expectedBias, 6);
});

test("neuron throws error when updating weights without forward pass", () => {
    const neuron = new Neuron(2);
    
    expect(() => neuron.updateWeights(0.1, 0.1)).toThrow('Cannot update weights: no last inputs data available');
});

test("neuron getWeights and setWeights work correctly", () => {
    const neuron = new Neuron(3);
    const newWeights = [0.1, 0.2, 0.3];
    const newBias = 0.4;
    
    neuron.setWeights(newWeights, newBias);
    
    const weights = neuron.getWeights();
    expect(weights.weights).toEqual(newWeights);
    expect(weights.bias).toBe(newBias);
    
    // Test immutability
    weights.weights[0] = 999;
    expect(neuron.weights[0]).toBe(0.1);
});

test("neuron provides correct information", () => {
    const neuron = new Neuron(2);
    neuron.setWeights([0.5, -0.3], 0.1);
    
    const info = neuron.getInfo();
    
    expect(info.inputSize).toBe(2);
    expect(info.weightsSum).toBeCloseTo(0.8, 6); // |0.5| + |-0.3| = 0.8
    expect(info.bias).toBe(0.1);
    expect(info.lastOutput).toBeNull();
    
    // After forward pass
    const sigmoid = (x) => 1 / (1 + Math.exp(-x));
    const output = neuron.forward([1, 1], sigmoid);
    
    const updatedInfo = neuron.getInfo();
    expect(updatedInfo.lastOutput).toBeCloseTo(output, 6);
});

test("neuron handles edge cases in weight calculation", () => {
    const neuron = new Neuron(1);
    neuron.setWeights([2.0], -1.0);
    
    const linear = (x) => x;
    const inputs = [0.5];
    
    const output = neuron.forward(inputs, linear);
    
    // Expected: 0.5 * 2.0 + (-1.0) = 0.0
    expect(output).toBeCloseTo(0.0, 6);
});

test("neuron maintains numerical stability with large values", () => {
    const neuron = new Neuron(2);
    neuron.setWeights([100, -100], 0);
    
    const sigmoid = (x) => 1 / (1 + Math.exp(-x));
    const inputs = [0.01, 0.01];
    
    const output = neuron.forward(inputs, sigmoid);
    
    // Should handle large intermediate values gracefully
    expect(output).toBeGreaterThanOrEqual(0);
    expect(output).toBeLessThanOrEqual(1);
    expect(isFinite(output)).toBe(true);
});
