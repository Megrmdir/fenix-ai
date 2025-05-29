import { test, expect } from "bun:test";
import { Layer } from "../src/core/Layer.js";
import { activations } from "../src/activations/index.js";

test("layer создается с правильным количеством нейронов", () => {
    const layer = new Layer(3, 2, activations.relu);
    
    expect(layer.neurons).toHaveLength(3);
    expect(layer.size).toBe(3);
    expect(layer.inputSize).toBe(2);
});

test("layer выполняет прямое распространение", () => {
    const layer = new Layer(2, 3, activations.sigmoid);
    const inputs = [0.5, -0.2, 0.8];
    
    const outputs = layer.forward(inputs);
    
    expect(outputs).toHaveLength(2);
    outputs.forEach(output => {
        expect(output).toBeGreaterThan(0);
        expect(output).toBeLessThan(1);
    });
});
