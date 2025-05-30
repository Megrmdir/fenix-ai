import { test, expect } from "bun:test";
import { Network } from "../src/core/Network.js";

/**
 * Test suite for Network class
 * Validates complete neural network functionality including training, prediction, and model management
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
    },
    linear: {
        func: (x) => x,
        derivative: (y) => 1,
        name: 'linear'
    }
};

test("network initializes correctly", () => {
    const network = new Network();

    expect(network.layers).toEqual([]);
    expect(network.learningRate).toBe(0.1);
    expect(network.trainingHistory).toEqual([]);
    expect(network.isCompiled).toBe(false);
});

test("network adds layers correctly", () => {
    const network = new Network();

    network.addLayer(4, activations.relu, 2)
        .addLayer(3, activations.tanh)
        .addLayer(1, activations.sigmoid);

    expect(network.layers).toHaveLength(3);
    expect(network.layers[0].size).toBe(4);
    expect(network.layers[0].inputSize).toBe(2);
    expect(network.layers[1].size).toBe(3);
    expect(network.layers[1].inputSize).toBe(4);
    expect(network.layers[2].size).toBe(1);
    expect(network.layers[2].inputSize).toBe(3);
    expect(network.isCompiled).toBe(false);
});

test("network validates layer parameters", () => {
    const network = new Network();

    // First layer must have input size
    expect(() => network.addLayer(2, activations.sigmoid)).toThrow('Input size must be specified for the first layer');

    // Invalid neuron count
    expect(() => network.addLayer(0, activations.sigmoid, 2)).toThrow('Number of neurons must be a positive integer');
    expect(() => network.addLayer(-1, activations.sigmoid, 2)).toThrow('Number of neurons must be a positive integer');

    // Invalid activation function
    expect(() => network.addLayer(2, {}, 2)).toThrow('Activation function must be an object with func method');
});

test("network compiles correctly", () => {
    const network = new Network();

    // Cannot compile empty network
    expect(() => network.compile()).toThrow('Network must contain at least one layer');

    // Can compile with layers
    network.addLayer(2, activations.sigmoid, 1);
    network.compile();
    expect(network.isCompiled).toBe(true);
});

test("network makes predictions correctly", () => {
    const network = new Network();
    network.addLayer(2, activations.sigmoid, 2)
        .addLayer(1, activations.sigmoid);

    const prediction = network.predict([0.5, 0.8]);

    expect(prediction).toHaveLength(1);
    expect(prediction[0]).toBeGreaterThanOrEqual(0);
    expect(prediction[0]).toBeLessThanOrEqual(1);
    expect(isFinite(prediction[0])).toBe(true);
});

test("network validates prediction inputs", () => {
    const network = new Network();
    network.addLayer(2, activations.sigmoid, 3);

    // Invalid input type
    expect(() => network.predict("invalid")).toThrow('Input data must be an array');

    // Invalid input size
    expect(() => network.predict([1, 2])).toThrow('Input size (2) does not match expected size (3)');
    expect(() => network.predict([1, 2, 3, 4])).toThrow('Input size (4) does not match expected size (3)');

    // Empty network
    const emptyNetwork = new Network();
    expect(() => emptyNetwork.predict([1])).toThrow('Network contains no layers');
});

test("network trains successfully on XOR problem", () => {
    const network = new Network();
    network.addLayer(4, activations.tanh, 2)
        .addLayer(1, activations.sigmoid)
        .setLearningRate(0.3);

    const xorData = [
        { input: [0, 0], target: [0] },
        { input: [0, 1], target: [1] },
        { input: [1, 0], target: [1] },
        { input: [1, 1], target: [0] }
    ];

    const initialError = calculateNetworkError(network, xorData);
    network.train(xorData, 1000, { verbose: false });
    const finalError = calculateNetworkError(network, xorData);

    expect(finalError).toBeLessThan(initialError);
    expect(network.trainingHistory).toHaveLength(1000);

    // Test predictions
    const tolerance = 0.3;
    expect(network.predict([0, 0])[0]).toBeLessThan(tolerance);
    expect(network.predict([0, 1])[0]).toBeGreaterThan(1 - tolerance);
    expect(network.predict([1, 0])[0]).toBeGreaterThan(1 - tolerance);
    expect(network.predict([1, 1])[0]).toBeLessThan(tolerance);
});

test("network validates training data", () => {
    const network = new Network();
    network.addLayer(1, activations.sigmoid, 2);

    // Empty training data
    expect(() => network.train([])).toThrow('Training data must be a non-empty array');
    expect(() => network.train("invalid")).toThrow('Training data must be a non-empty array');

    // Invalid example format
    expect(() => network.train([{ input: [1, 2] }])).toThrow('Each example must contain input and target fields');
    expect(() => network.train([{ target: [1] }])).toThrow('Each example must contain input and target fields');

    // Inconsistent sizes
    const inconsistentData = [
        { input: [1, 2], target: [0] },
        { input: [1], target: [0] }  // Different input size
    ];
    expect(() => network.train(inconsistentData)).toThrow('All input vectors must have the same size');

    const inconsistentTargets = [
        { input: [1, 2], target: [0] },
        { input: [1, 2], target: [0, 1] }  // Different target size
    ];
    expect(() => network.train(inconsistentTargets)).toThrow('All target vectors must have the same size');
});

test("network handles learning rate changes", () => {
    const network = new Network();

    // Valid learning rate
    network.setLearningRate(0.05);
    expect(network.learningRate).toBe(0.05);

    // Invalid learning rates
    expect(() => network.setLearningRate(0)).toThrow('Learning rate must be a positive number');
    expect(() => network.setLearningRate(-0.1)).toThrow('Learning rate must be a positive number');
    expect(() => network.setLearningRate("invalid")).toThrow('Learning rate must be a positive number');
});

test("network provides correct information", () => {
    const network = new Network();
    network.addLayer(4, activations.relu, 2)
        .addLayer(3, activations.tanh)
        .addLayer(1, activations.sigmoid)
        .setLearningRate(0.05);

    const info = network.getInfo();

    expect(info.layers).toBe(3);
    expect(info.architecture).toEqual([4, 3, 1]);
    expect(info.inputSize).toBe(2);
    expect(info.outputSize).toBe(1);
    expect(info.totalParameters).toBe(4 * 2 + 4 + 3 * 4 + 3 + 1 * 3 + 1); // weights + biases
    expect(info.learningRate).toBe(0.05);
    expect(info.isCompiled).toBe(false);
    expect(info.trainedEpochs).toBe(0);
    expect(info.lastError).toBeNull();

    // After training
    const data = [{ input: [1, 1], target: [0.5] }];
    network.train(data, 10);

    const updatedInfo = network.getInfo();
    expect(updatedInfo.trainedEpochs).toBe(10);
    expect(updatedInfo.lastError).toBeTypeOf('number');
});

test("network exports and handles model data", () => {
    const network = new Network();
    network.addLayer(2, activations.sigmoid, 1)
        .addLayer(1, activations.linear)
        .setLearningRate(0.2);

    // Train briefly to have some history
    const data = [{ input: [0.5], target: [0.8] }];
    network.train(data, 5);

    const exportedModel = network.exportModel();
    const modelData = JSON.parse(exportedModel);

    expect(modelData.architecture).toHaveLength(2);
    expect(modelData.learningRate).toBe(0.2);
    expect(modelData.trainingHistory).toHaveLength(5);

    // Check architecture structure
    expect(modelData.architecture[0].neuronCount).toBe(2);
    expect(modelData.architecture[0].inputSize).toBe(1);
    expect(modelData.architecture[1].neuronCount).toBe(1);
    expect(modelData.architecture[1].inputSize).toBe(2);
});

test("network resets correctly", () => {
    const network = new Network();
    network.addLayer(1, activations.sigmoid, 1);
    
    // Train and make predictions
    const data = [{ input: [0.5], target: [0.8] }];
    network.train(data, 10);
    network.predict([0.5]);
    
    expect(network.trainingHistory).toHaveLength(10);
    expect(network.layers[0].outputs).not.toEqual([]);
    
    // Reset
    network.reset();
    
    expect(network.trainingHistory).toEqual([]);
    expect(network.layers[0].outputs).toEqual([]);
    expect(network.layers[0].neurons[0].lastInputs).toBeNull();
});

test("network handles training with validation and early stopping", () => {
    const network = new Network();
    network.addLayer(3, activations.tanh, 2)
        .addLayer(1, activations.sigmoid);

    const trainingData = [
        { input: [0, 0], target: [0] },
        { input: [1, 1], target: [0] }
    ];

    const validationData = [
        { input: [0, 1], target: [1] },
        { input: [1, 0], target: [1] }
    ];

    network.train(trainingData, 1000, {
        validationData,
        earlyStoppingPatience: 50,
        verbose: false
    });

    // Should stop before 1000 epochs due to early stopping
    expect(network.trainingHistory.length).toBeLessThan(1000);
});

test("network handles regression task", () => {
    const network = new Network();
    network.addLayer(5, activations.relu, 1)
        .addLayer(1, activations.linear)
        .setLearningRate(0.01);

    // Generate y = x^2 data
    const data = [];
    for (let i = 0; i < 20; i++) {
        const x = (i / 10) - 1; // x from -1 to 1
        data.push({ input: [x], target: [x * x] });
    }

    const initialError = calculateNetworkError(network, data);
    network.train(data, 500);
    const finalError = calculateNetworkError(network, data);

    expect(finalError).toBeLessThan(initialError);

    // Test specific predictions
    const pred1 = network.predict([0.5])[0];
    const pred2 = network.predict([-0.3])[0];

    expect(pred1).toBeCloseTo(0.25, 0); // Точность до 0.5
    expect(pred2).toBeCloseTo(0.09, 0); // Точность до 0.5
});

test("network training history is accessible", () => {
    const network = new Network();
    network.addLayer(1, activations.sigmoid, 1);

    const data = [{ input: [0.5], target: [0.8] }];
    network.train(data, 5);

    const history = network.getTrainingHistory();

    expect(history).toHaveLength(5);
    expect(history).toEqual(network.trainingHistory);

    // Test immutability
    history[0] = 999;
    expect(network.trainingHistory[0]).not.toBe(999);
});

// Helper function to calculate network error
function calculateNetworkError(network, data) {
    let totalError = 0;
    for (const example of data) {
        const prediction = network.predict(example.input);
        const error = example.target.reduce((sum, target, i) => {
            return sum + Math.pow(target - prediction[i], 2);
        }, 0);
        totalError += error;
    }
    return totalError / data.length;
}
