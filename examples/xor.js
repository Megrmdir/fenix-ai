import { Network } from "../src/core/Network.js";
import { activations } from "../src/activations/index.js";

console.log("Fenix.AI - Решение XOR");

// Создаем сеть
const network = new Network();

// Архитектура: 2 входа -> 4 скрытых нейрона -> 1 выход
network
    .addLayer(2, activations.sin, 2)    // Скрытый слой: 2 нейрона
    .addLayer(1, activations.sigmoid);   // Выходной слой: 1 нейрон, Sigmoid

// Устанавливаем скорость обучения
network.setLearningRate(0.07);

// Данные XOR
const xorData = [
    { input: [0, 0], target: [0] },  // 0 XOR 0 = 0
    { input: [0, 1], target: [1] },  // 0 XOR 1 = 1
    { input: [1, 0], target: [1] },  // 1 XOR 0 = 1
    { input: [1, 1], target: [0] }   // 1 XOR 1 = 0
];

console.log("архитектура сети:", network.getInfo());

// Тестируем до обучения
console.log("\nрезультаты ДО обучения:");
xorData.forEach(example => {
    const prediction = network.predict(example.input);
    console.log(`[${example.input}] -> ${prediction[0].toFixed(4)} (ожидалось: ${example.target[0]})`);
});

// Обучаем сеть
network.train(xorData, 200000);

// Тестируем после обучения
console.log("\n🎯 результаты ПОСЛЕ обучения:");
xorData.forEach(example => {
    const prediction = network.predict(example.input);
    const isCorrect = Math.abs(prediction[0] - example.target[0]) < 0.1;
    console.log(`[${example.input}] -> ${prediction[0].toFixed(4)} (ожидалось: ${example.target[0]}) ${isCorrect ? '✅' : '❌'}`);
});

console.log("\nXOR решена");
