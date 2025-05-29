import { Neuron } from "../src/core/Neuron.js";
import { activations } from "../src/activations/index.js";

console.log("Fenix.AI - тест нейрона");

// Создаем нейрон с 2 входами
const neuron = new Neuron(2);

console.log("начальные веса:", neuron.getWeights());

// Тестируем разные входы
const testInputs = [
    [1, 0],
    [0, 1], 
    [1, 1],
    [0, 0]
];

console.log("\nтестируем нейрон:");
testInputs.forEach((inputs, i) => {
    const output = neuron.forward(inputs, activations.sigmoid.func);
    console.log(`вход: [${inputs.join(', ')}] -> выход: ${output.toFixed(4)}`);
});

console.log("\nнейрон работает");
