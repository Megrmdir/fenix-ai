import { test, expect } from "bun:test";
import { activations } from "../src/activations/index.js";

/**
 * Test suite for Activation Functions
 * Validates mathematical correctness and edge cases for all activation functions
 */

test("sigmoid activation function works correctly", () => {
    // Test basic functionality
    expect(activations.sigmoid.func(0)).toBeCloseTo(0.5, 6);
    expect(activations.sigmoid.func(1)).toBeCloseTo(0.7311, 4);
    expect(activations.sigmoid.func(-1)).toBeCloseTo(0.2689, 4);
    
    // Test extreme values (with clamping)
    expect(activations.sigmoid.func(1000)).toBeCloseTo(1.0, 6);
    expect(activations.sigmoid.func(-1000)).toBeCloseTo(0.0, 6);
    
    // Test derivative
    const output1 = activations.sigmoid.func(0);
    expect(activations.sigmoid.derivative(output1)).toBeCloseTo(0.25, 6); // 0.5 * (1 - 0.5) = 0.25
    
    const output2 = activations.sigmoid.func(2);
    expect(activations.sigmoid.derivative(output2)).toBeCloseTo(output2 * (1 - output2), 6);
});

test("relu activation function works correctly", () => {
    // Test basic functionality
    expect(activations.relu.func(0)).toBe(0);
    expect(activations.relu.func(5)).toBe(5);
    expect(activations.relu.func(-3)).toBe(0);
    expect(activations.relu.func(0.5)).toBe(0.5);
    expect(activations.relu.func(-0.1)).toBe(0);
    
    // Test extreme values
    expect(activations.relu.func(1000)).toBe(1000);
    expect(activations.relu.func(-1000)).toBe(0);
    
    // Test derivative
    expect(activations.relu.derivative(5)).toBe(1);
    expect(activations.relu.derivative(0)).toBe(0);
    expect(activations.relu.derivative(-3)).toBe(0);
    expect(activations.relu.derivative(0.1)).toBe(1);
});

test("leaky relu activation function works correctly", () => {
    const alpha = 0.01;
    
    // Test basic functionality
    expect(activations.leakyRelu.func(0)).toBe(0);
    expect(activations.leakyRelu.func(5)).toBe(5);
    expect(activations.leakyRelu.func(-2)).toBeCloseTo(-2 * alpha, 6);
    expect(activations.leakyRelu.func(0.5)).toBe(0.5);
    expect(activations.leakyRelu.func(-0.1)).toBeCloseTo(-0.1 * alpha, 6);
    
    // Test derivative
    expect(activations.leakyRelu.derivative(5)).toBe(1);
    expect(activations.leakyRelu.derivative(0)).toBe(alpha);
    expect(activations.leakyRelu.derivative(-3)).toBe(alpha);
    expect(activations.leakyRelu.derivative(0.1)).toBe(1);
});

test("elu activation function works correctly", () => {
    // Test basic functionality
    expect(activations.elu.func(0)).toBe(0);
    expect(activations.elu.func(5)).toBe(5);
    expect(activations.elu.func(-1)).toBeCloseTo(Math.exp(-1) - 1, 6);
    expect(activations.elu.func(0.5)).toBe(0.5);
    
    // Test negative values approach -1 (reduced precision)
    expect(activations.elu.func(-10)).toBeCloseTo(-1, 4);
    
    // Test derivative
    expect(activations.elu.derivative(5)).toBe(1);
    expect(activations.elu.derivative(0)).toBe(1);
    expect(activations.elu.derivative(-1)).toBeCloseTo(Math.exp(-1), 6);
});

test("tanh activation function works correctly", () => {
    // Test basic functionality
    expect(activations.tanh.func(0)).toBeCloseTo(0, 6);
    expect(activations.tanh.func(1)).toBeCloseTo(Math.tanh(1), 6);
    expect(activations.tanh.func(-1)).toBeCloseTo(Math.tanh(-1), 6);
    
    // Test extreme values
    expect(activations.tanh.func(100)).toBeCloseTo(1.0, 6);
    expect(activations.tanh.func(-100)).toBeCloseTo(-1.0, 6);
    
    // Test derivative: d/dx tanh(x) = 1 - tanh²(x)
    const output1 = activations.tanh.func(0);
    expect(activations.tanh.derivative(output1)).toBeCloseTo(1, 6); // 1 - 0² = 1
    
    const output2 = activations.tanh.func(1);
    expect(activations.tanh.derivative(output2)).toBeCloseTo(1 - output2 * output2, 6);
});

test("swish activation function works correctly", () => {
    // Test basic functionality
    expect(activations.swish.func(0)).toBeCloseTo(0, 6);
    expect(activations.swish.func(1)).toBeCloseTo(1 / (1 + Math.exp(-1)), 6);
    expect(activations.swish.func(-1)).toBeCloseTo(-1 / (1 + Math.exp(1)), 6);
    
    // Test that swish(x) = x * sigmoid(x)
    const x = 2.5;
    const sigmoid = 1 / (1 + Math.exp(-x));
    expect(activations.swish.func(x)).toBeCloseTo(x * sigmoid, 6);
    
    // Test derivative
    const x2 = 1.0;
    const sig = 1 / (1 + Math.exp(-x2));
    const expectedDerivative = sig + x2 * sig * (1 - sig);
    expect(activations.swish.derivative(x2)).toBeCloseTo(expectedDerivative, 6);
});

test("mish activation function works correctly", () => {
    // Test basic functionality
    expect(activations.mish.func(0)).toBeCloseTo(0, 6);
    expect(activations.mish.func(1)).toBeGreaterThan(0);
    expect(activations.mish.func(-1)).toBeLessThan(0);
    
    // Test that mish is approximately x * tanh(softplus(x))
    const x = 2.0;
    const softplus = Math.log(1 + Math.exp(x));
    const expectedMish = x * Math.tanh(softplus);
    expect(activations.mish.func(x)).toBeCloseTo(expectedMish, 6);
    
    // Test derivative exists and is finite
    expect(isFinite(activations.mish.derivative(1.0))).toBe(true);
    expect(isFinite(activations.mish.derivative(-1.0))).toBe(true);
});

test("sinusoidal activation function works correctly", () => {
    // Test basic functionality
    expect(activations.sin.func(0)).toBeCloseTo(0, 6);
    expect(activations.sin.func(Math.PI / 2)).toBeCloseTo(1, 6);
    expect(activations.sin.func(Math.PI)).toBeCloseTo(0, 6);
    expect(activations.sin.func(-Math.PI / 2)).toBeCloseTo(-1, 6);
    
    // Test derivative: d/dx sin(x) = cos(x)
    expect(activations.sin.derivative(0)).toBeCloseTo(Math.cos(0), 6);
    expect(activations.sin.derivative(Math.PI / 4)).toBeCloseTo(Math.cos(Math.PI / 4), 6);
    expect(activations.sin.derivative(Math.PI / 2)).toBeCloseTo(Math.cos(Math.PI / 2), 6);
});

test("cosine activation function works correctly", () => {
    // Test basic functionality
    expect(activations.cos.func(0)).toBeCloseTo(1, 6);
    expect(activations.cos.func(Math.PI / 2)).toBeCloseTo(0, 6);
    expect(activations.cos.func(Math.PI)).toBeCloseTo(-1, 6);
    expect(activations.cos.func(-Math.PI / 2)).toBeCloseTo(0, 6);
    
    // Test derivative: d/dx cos(x) = -sin(x)
    expect(activations.cos.derivative(0)).toBeCloseTo(-Math.sin(0), 6);
    expect(activations.cos.derivative(Math.PI / 4)).toBeCloseTo(-Math.sin(Math.PI / 4), 6);
    expect(activations.cos.derivative(Math.PI / 2)).toBeCloseTo(-Math.sin(Math.PI / 2), 6);
});

test("gaussian activation function works correctly", () => {
    // Test basic functionality
    expect(activations.gaussian.func(0)).toBeCloseTo(1, 6); // exp(-0²) = 1
    expect(activations.gaussian.func(1)).toBeCloseTo(Math.exp(-1), 6);
    expect(activations.gaussian.func(-1)).toBeCloseTo(Math.exp(-1), 6);
    
    // Test symmetry
    expect(activations.gaussian.func(2)).toBeCloseTo(activations.gaussian.func(-2), 6);
    
    // Test derivative: d/dx exp(-x²) = -2x * exp(-x²)
    const x = 1.5;
    const expectedDerivative = -2 * x * Math.exp(-x * x);
    expect(activations.gaussian.derivative(x)).toBeCloseTo(expectedDerivative, 6);
    
    // Test derivative at zero
    expect(activations.gaussian.derivative(0)).toBeCloseTo(0, 6);
});

test("step activation function works correctly", () => {
    // Test basic functionality
    expect(activations.step.func(0)).toBe(0);
    expect(activations.step.func(0.1)).toBe(1);
    expect(activations.step.func(5)).toBe(1);
    expect(activations.step.func(-0.1)).toBe(0);
    expect(activations.step.func(-5)).toBe(0);
    
    // Test derivative (always 0 for step function)
    expect(activations.step.derivative(5)).toBe(0);
    expect(activations.step.derivative(0)).toBe(0);
    expect(activations.step.derivative(-3)).toBe(0);
});

test("linear activation function works correctly", () => {
    // Test basic functionality
    expect(activations.linear.func(0)).toBe(0);
    expect(activations.linear.func(5)).toBe(5);
    expect(activations.linear.func(-3)).toBe(-3);
    expect(activations.linear.func(0.5)).toBe(0.5);
    expect(activations.linear.func(-0.1)).toBe(-0.1);
    
    // Test extreme values
    expect(activations.linear.func(1000)).toBe(1000);
    expect(activations.linear.func(-1000)).toBe(-1000);
    
    // Test derivative (always 1 for linear)
    expect(activations.linear.derivative(5)).toBe(1);
    expect(activations.linear.derivative(0)).toBe(1);
    expect(activations.linear.derivative(-3)).toBe(1);
    expect(activations.linear.derivative(1000)).toBe(1);
});

test("activation functions handle edge cases correctly", () => {
    // Test with very small numbers
    expect(activations.sigmoid.func(1e-10)).toBeCloseTo(0.5, 6);
    expect(activations.tanh.func(1e-10)).toBeCloseTo(0, 6);
    expect(activations.relu.func(1e-10)).toBe(1e-10);
    expect(activations.leakyRelu.func(1e-10)).toBe(1e-10);
    expect(activations.linear.func(1e-10)).toBe(1e-10);
    
    // Test with negative small numbers
    expect(activations.sigmoid.func(-1e-10)).toBeCloseTo(0.5, 6);
    expect(activations.tanh.func(-1e-10)).toBeCloseTo(0, 6);
    expect(activations.relu.func(-1e-10)).toBe(0);
    expect(activations.leakyRelu.func(-1e-10)).toBeCloseTo(-1e-10 * 0.01, 12);
    expect(activations.linear.func(-1e-10)).toBe(-1e-10);
});

test("activation functions maintain mathematical properties", () => {
    // Sigmoid: output always between 0 and 1
    const sigmoidValues = [-10, -1, 0, 1, 10].map(x => activations.sigmoid.func(x));
    sigmoidValues.forEach(val => {
        expect(val).toBeGreaterThan(0);
        expect(val).toBeLessThan(1);
    });
    
    // Tanh: output always between -1 and 1
    const tanhValues = [-10, -1, 0, 1, 10].map(x => activations.tanh.func(x));
    tanhValues.forEach(val => {
        expect(val).toBeGreaterThanOrEqual(-1);
        expect(val).toBeLessThanOrEqual(1);
    });
    
    // ReLU: non-negative outputs
    const reluValues = [-10, -1, 0, 1, 10].map(x => activations.relu.func(x));
    reluValues.forEach(val => {
        expect(val).toBeGreaterThanOrEqual(0);
    });
    
    // Gaussian: always positive, max at x=0
    expect(activations.gaussian.func(0)).toBeGreaterThan(activations.gaussian.func(1));
    expect(activations.gaussian.func(0)).toBeGreaterThan(activations.gaussian.func(-1));
    expect(activations.gaussian.func(1)).toBeCloseTo(activations.gaussian.func(-1), 6);
    
    // Step: only 0 or 1
    const stepValues = [-5, -0.1, 0, 0.1, 5].map(x => activations.step.func(x));
    stepValues.forEach(val => {
        expect([0, 1]).toContain(val);
    });
});

test("trigonometric functions maintain periodicity", () => {
    // Sin function periodicity
    const x = Math.PI / 3;
    expect(activations.sin.func(x)).toBeCloseTo(activations.sin.func(x + 2 * Math.PI), 6);
    expect(activations.sin.func(x)).toBeCloseTo(activations.sin.func(x + 4 * Math.PI), 6);
    
    // Cos function periodicity
    expect(activations.cos.func(x)).toBeCloseTo(activations.cos.func(x + 2 * Math.PI), 6);
    expect(activations.cos.func(x)).toBeCloseTo(activations.cos.func(x + 4 * Math.PI), 6);
    
    // Sin and Cos relationship
    expect(activations.sin.func(x)).toBeCloseTo(activations.cos.func(Math.PI / 2 - x), 6);
});

test("advanced activation functions numerical stability", () => {
    // Test Swish with large values
    const largeValue = 100;
    const swishLarge = activations.swish.func(largeValue);
    expect(isFinite(swishLarge)).toBe(true);
    expect(swishLarge).toBeCloseTo(largeValue, 1); // Should approach x for large x
    
    // Test Mish with large values
    const mishLarge = activations.mish.func(largeValue);
    expect(isFinite(mishLarge)).toBe(true);
    expect(mishLarge).toBeCloseTo(largeValue, 1); // Should approach x for large x
    
    // Test ELU with very negative values
    const eluVeryNegative = activations.elu.func(-100);
    expect(eluVeryNegative).toBeCloseTo(-1, 6); // Should approach -1 for very negative x
});

test("all activation functions have required interface", () => {
    const activationNames = [
        'sigmoid', 'relu', 'leakyRelu', 'elu', 'tanh', 'swish', 
        'mish', 'sin', 'cos', 'gaussian', 'step', 'linear'
    ];
    
    activationNames.forEach(name => {
        expect(activations).toHaveProperty(name);
        expect(activations[name]).toHaveProperty('func');
        expect(activations[name]).toHaveProperty('derivative');
        expect(typeof activations[name].func).toBe('function');
        expect(typeof activations[name].derivative).toBe('function');
    });
});

test("derivative consistency check", () => {
    // Test that derivatives return reasonable values for common inputs
    const testInputs = [-2, -1, -0.5, 0, 0.5, 1, 2];
    
    testInputs.forEach(input => {
        // All derivatives should be finite
        expect(isFinite(activations.sigmoid.derivative(activations.sigmoid.func(input)))).toBe(true);
        expect(isFinite(activations.tanh.derivative(activations.tanh.func(input)))).toBe(true);
        expect(isFinite(activations.swish.derivative(input))).toBe(true);
        expect(isFinite(activations.mish.derivative(input))).toBe(true);
        expect(isFinite(activations.sin.derivative(input))).toBe(true);
        expect(isFinite(activations.cos.derivative(input))).toBe(true);
        expect(isFinite(activations.gaussian.derivative(input))).toBe(true);
        
        // Linear derivative is always 1
        expect(activations.linear.derivative(input)).toBe(1);
        
        // Step derivative is always 0
        expect(activations.step.derivative(input)).toBe(0);
    });
});
