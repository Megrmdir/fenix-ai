# Fenix.AI

Простая библиотека нейронных сетей на чистом JavaScript без зависимостей.

## Установка

```bash
git clone https://github.com/megrmdir/fenix-ai.git
cd fenix-ai
```

## Быстрый старт

```javascript
import { Network } from "./src/core/Network.js";
import { activations } from "./src/activations/index.js";

// Создаем сеть
const net = new Network();
net.addLayer(2, activations.tanh, 2)
   .addLayer(1, activations.sigmoid)
   .setLearningRate(0.1);

// Данные XOR
const data = [
    { input: [0, 0], target: [0] },
    { input: [0, 1], target: [1] },
    { input: [1, 0], target: [1] },
    { input: [1, 1], target: [0] }
];

// Обучение
net.train(data, 5000);

// Тестирование
data.forEach(({input, target}) => {
    const result = net.predict(input);
    console.log(`${input} -> ${result.toFixed(3)} (ожидалось: ${target})`);
});
```

## Структура проекта

```text
fenix-ai/
├── src/
│   ├── core/
│   │   ├── Neuron.js
│   │   ├── Layer.js
│   │   └── Network.js
│   └── activations/
│       └── index.js
├── examples/
│   └── xor.js
└── tests/
    ├── neuron.test.js
    ├── layer.test.js
    └── network.test.js
```

## API

### Network

```javascript
const net = new Network();

// Добавить слой
net.addLayer(количество_нейронов, функция_активации, размер_входа);

// Установить скорость обучения
net.setLearningRate(0.1);

// Обучить сеть
net.train(данные, количество_эпох);

// Сделать предсказание
const результат = net.predict([0.5, 0.8]);

// Получить информацию о сети
console.log(net.getInfo());
```

#### addLayer(neuronCount, activation, inputSize)

Добавляет новый слой в сеть.

**Параметры:**
- `neuronCount` - количество нейронов в слое
- `activation` - функция активации из объекта activations
- `inputSize` - размер входного вектора (только для первого слоя)

```javascript
net.addLayer(4, activations.relu, 2);  // Первый слой
net.addLayer(1, activations.sigmoid);  // Выходной слой
```

#### train(data, epochs)

Обучает сеть на предоставленных данных.

**Параметры:**
- `data` - массив объектов с полями input и target
- `epochs` - количество эпох обучения

```javascript
const trainingData = [
    { input: [0.1, 0.2], target: [0.8] },
    { input: [0.4, 0.5], target: [0.2] }
];
net.train(trainingData, 1000);
```

#### predict(input)

Выполняет предсказание для входного вектора.

```javascript
const result = net.predict([0.5, 0.8]);
console.log(result); // [0.7234]
```

### Функции активации

```javascript
import { activations } from "./src/activations/index.js";

activations.sigmoid    // Сигмоида (0, 1)
activations.tanh       // Тангенс (-1, 1)
activations.relu       // ReLU (0, ∞)
activations.linear     // Линейная
activations.leakyRelu  // Leaky ReLU
activations.swish      // Swish
activations.mish       // Mish
activations.softplus   // Softplus
activations.elu        // ELU
```

## Примеры

### Бинарная классификация

```javascript
const net = new Network();
net.addLayer(4, activations.relu, 2)
   .addLayer(1, activations.sigmoid)
   .setLearningRate(0.1);

const data = [
    { input: [0.2, 0.3], target: [0] },  // Класс 0
    { input: [0.7, 0.8], target: [1] },  // Класс 1
    { input: [0.1, 0.4], target: [0] },  // Класс 0
    { input: [0.6, 0.9], target: [1] }   // Класс 1
];

net.train(data, 2000);

console.log(net.predict([0.4, 0.3])); // Близко к [0]
console.log(net.predict([0.8, 0.9])); // Близко к [1]
```

### Регрессия

```javascript
const net = new Network();
net.addLayer(8, activations.relu, 1)
   .addLayer(1, activations.linear)
   .setLearningRate(0.01);

// Аппроксимация y = x²
const data = [];
for (let i = 0; i < 50; i++) {
    const x = i / 25 - 1;  // x от -1 до 1
    data.push({ input: [x], target: [x * x] });
}

net.train(data, 3000);

console.log(net.predict([0.5]));   // Должно быть близко к [0.25]
console.log(net.predict([-0.3]));  // Должно быть близко к [0.09]

// Нормализация данных

// В диапазон [0, 1]
function normalize(data) {
    const max = Math.max(...data);
    const min = Math.min(...data);
    return data.map(x => (x - min) / (max - min));
}

// В диапазон [-1, 1]
function normalizeSymmetric(data) {
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min;
    return data.map(x => 2 * (x - min) / range - 1);
}

// Z-score
function zScore(data) {
    const mean = data.reduce((a, b) => a + b) / data.length;
    const std = Math.sqrt(data.reduce((a, b) => a + (b - mean) ** 2, 0) / data.length);
    return data.map(x => (x - mean) / std);
}
```

## Отладка

### Проверка ошибки

```javascript
function checkError(net, data) {
    let totalError = 0;
    for (const {input, target} of data) {
        const pred = net.predict(input);
        const error = target.reduce((sum, t, i) => sum + (t - pred[i]) ** 2, 0);
        totalError += error;
    }
    return totalError / data.length;
}

const beforeError = checkError(net, data);
net.train(data, 1000);
const afterError = checkError(net, data);
console.log(`Улучшение: ${((beforeError - afterError) / beforeError * 100).toFixed(1)}%`);
```

### Просмотр весов

```javascript
function printWeights(net) {
    net.layers.forEach((layer, i) => {
        console.log(`Слой ${i + 1}:`);
        layer.neurons.forEach((neuron, j) => {
            const {weights, bias} = neuron.getWeights();
            console.log(`  Нейрон ${j + 1}: [${weights.map(w => w.toFixed(2))}] bias: ${bias.toFixed(2)}`);
        });
    });
}
```

## Тестирование

```bash
# Все тесты
bun test

# Конкретный тест
bun test tests/neuron.test.js

# Примеры
bun run examples/xor.js
```

### Создание тестов

```javascript
import { test, expect } from "bun:test";
import { Network } from "../src/core/Network.js";
import { activations } from "../src/activations/index.js";

test("сеть обучается", () => {
    const net = new Network();
    net.addLayer(2, activations.relu, 1)
       .addLayer(1, activations.sigmoid);
    
    const data = [{ input: [0.5], target: [0.8] }];
    net.train(data, 100);
    
    const result = net.predict([0.5]);
    expect(result).toBeCloseTo(0.8, 1);
});
```

## Производительность

**Время обучения:**
- Сеть 10-5-1, 1K примеров, 1K эпох: ~1-2 сек
- Сеть 100-50-10, 1K примеров, 1K эпох: ~10-20 сек
- Сеть 1000-500-100, 10K примеров, 1K эпох: ~5-10 мин

**Оптимизация:**
1. Меньше слоев для простых задач
2. Нормализация входных данных
3. Правильная скорость обучения
4. ReLU быстрее sigmoid/tanh
5. Разумное количество эпох

## Ограничения

- Только полносвязные слои
- Только обратное распространение
- Нет регуляризации
- Нет CNN/RNN
- Только CPU
- Нет сохранения моделей

## Планы

- [ ] Регуляризация (dropout, L1/L2)
- [ ] Сохранение/загрузка моделей
- [ ] Больше оптимизаторов (Adam, RMSprop)
- [ ] Batch обучение
- [ ] Валидация и метрики
- [ ] Сверточные слои

## Сравнение
```
.                  |  Fenix.AI | TensorFlow.js |  Brain.js |
|------------------|-----------|---------------|-----------|
| Размер           |     ~10KB |        ~500KB |     ~50KB |
| Зависимости      |         - |   Очень много |         - |
| Простота         | ★★★★★★ |     ☆☆☆☆★★ | ☆☆★★★★ |
| Функциональность | ☆☆✫★★★ |     ★★★★★★ | ☆☆★★★★ |
| Скорость         | ☆☆✫★★★ |     ★★★★★★ | ☆☆★★★★ |
```
## FAQ

**Q: Работает в браузере?**  
A: Да, в любом современном браузере.

**Q: Поддержка TypeScript?**  
A: Пока нет, планируется.

**Q: Обучение на GPU?**  
A: Пока нет, планируется WebGL.

**Q: Сохранение модели?**  
A: Пока через JSON.stringify(), планируется полноценное.

## Лицензия

Этот проект лицензирован под Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

**Вы можете:**
- Использовать проект для обучения и некоммерческих целей
- Изменять и адаптировать код
- Делиться своими модификациями

**Вы обязаны:**
- Указывать авторство Tetra
- Отмечать, если вы внесли изменения
- Использовать ту же лицензию для производных работ
- **НЕ использовать проект в коммерческих целях**

Для коммерческого лицензирования обращайтесь по адресу megrm.dir@gmail.com.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License

## Поддержка

- [Issues](https://github.com/megrmdir/fenix-ai/issues) для багов
- Примеры в папке `examples/`
- Документация в комментариях кода

## Благодарности

Спасибо всем тестировщикам и контрибьюторам!
