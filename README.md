# Fenix.AI

A simple neural network library in pure JavaScript with no dependencies.

## Installation

```bash
git clone https://github.com/megrmdir/fenix-ai.git
cd fenix-ai
```

## Quick Start

```javascript
import { Network } from "./src/core/Network.js";
import { activations } from "./src/activations/index.js";

// Create network
const net = new Network();
net.addLayer(2, activations.tanh, 2)
   .addLayer(1, activations.sigmoid)
   .setLearningRate(0.1);

// XOR data
const data = [
    { input: [0, 0], target: [0] },
    { input: [0, 1], target: [1] },
    { input: [1, 0], target: [1] },
    { input: [1, 1], target: [0] }
];

// Training
net.train(data, 5000);

// Testing
data.forEach(({input, target}) => {
    const result = net.predict(input);
    console.log(`${input} -> ${result.toFixed(3)} (expected: ${target})`);
});
```

## Project Structure

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

// Add layer
net.addLayer(neuron_count, activation_function, input_size);

// Set learning rate
net.setLearningRate(0.1);

// Train network
net.train(data, epoch_count);

// Make prediction
const result = net.predict([0.5, 0.8]);

// Get network info
console.log(net.getInfo());
```

#### addLayer(neuronCount, activation, inputSize)

Adds a new layer to the network.

**Parameters:**
- `neuronCount` - number of neurons in the layer
- `activation` - activation function from activations object
- `inputSize` - input vector size (only for first layer)

```javascript
net.addLayer(4, activations.relu, 2);  // First layer
net.addLayer(1, activations.sigmoid);  // Output layer
```

#### train(data, epochs)

Trains the network on provided data.

**Parameters:**
- `data` - array of objects with input and target fields
- `epochs` - number of training epochs

```javascript
const trainingData = [
    { input: [0.1, 0.2], target: [0.8] },
    { input: [0.4, 0.5], target: [0.2] }
];
net.train(trainingData, 1000);
```

#### predict(input)

Performs prediction for input vector.

```javascript
const result = net.predict([0.5, 0.8]);
console.log(result); // [0.7234]
```

### Activation Functions

```javascript
import { activations } from "./src/activations/index.js";

activations.sigmoid    // Sigmoid (0, 1)
activations.tanh       // Tangent (-1, 1)
activations.relu       // ReLU (0, ∞)
activations.linear     // Linear
activations.leakyRelu  // Leaky ReLU
activations.swish      // Swish
activations.mish       // Mish
activations.softplus   // Softplus
activations.elu        // ELU
```

## Examples

### Binary Classification

```javascript
const net = new Network();
net.addLayer(4, activations.relu, 2)
   .addLayer(1, activations.sigmoid)
   .setLearningRate(0.1);

const data = [
    { input: [0.2, 0.3], target: [0] },  // Class 0
    { input: [0.7, 0.8], target: [1] },  // Class 1
    { input: [0.1, 0.4], target: [0] },  // Class 0
    { input: [0.6, 0.9], target: [1] }   // Class 1
];

net.train(data, 2000);

console.log(net.predict([0.4, 0.3])); // Close to [0]
console.log(net.predict([0.8, 0.9])); // Close to [1]
```

### Regression

```javascript
const net = new Network();
net.addLayer(8, activations.relu, 1)
   .addLayer(1, activations.linear)
   .setLearningRate(0.01);

// Approximating y = x²
const data = [];
for (let i = 0; i  (x - min) / (max - min));
}

// To range [-1, 1]
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

## Debugging

### Error Checking

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
console.log(`Improvement: ${((beforeError - afterError) / beforeError * 100).toFixed(1)}%`);
```

### Weight Inspection

```javascript
function printWeights(net) {
    net.layers.forEach((layer, i) => {
        console.log(`Layer ${i + 1}:`);
        layer.neurons.forEach((neuron, j) => {
            const {weights, bias} = neuron.getWeights();
            console.log(`  Neuron ${j + 1}: [${weights.map(w => w.toFixed(2))}] bias: ${bias.toFixed(2)}`);
        });
    });
}
```

## Testing

```bash
# All tests
bun test

# Specific test
bun test tests/neuron.test.js

# Examples
bun run examples/xor.js
```

### Creating Tests

```javascript
import { test, expect } from "bun:test";
import { Network } from "../src/core/Network.js";
import { activations } from "../src/activations/index.js";

test("network learns", () => {
    const net = new Network();
    net.addLayer(2, activations.relu, 1)
       .addLayer(1, activations.sigmoid);
    
    const data = [{ input: [0.5], target: [0.8] }];
    net.train(data, 100);
    
    const result = net.predict([0.5]);
    expect(result).toBeCloseTo(0.8, 1);
});
```

## Performance

**Training Time:**
- Network 10-5-1, 1K samples, 1K epochs: ~1-2 sec
- Network 100-50-10, 1K samples, 1K epochs: ~10-20 sec
- Network 1000-500-100, 10K samples, 1K epochs: ~5-10 min

**Optimization:**
1. Fewer layers for simple tasks
2. Input data normalization
3. Proper learning rate
4. ReLU faster than sigmoid/tanh
5. Reasonable number of epochs

## Limitations

- Only fully connected layers
- Only backpropagation
- No regularization
- No CNN/RNN
- CPU only
- No model saving

## Roadmap

- [ ] Regularization (dropout, L1/L2)
- [ ] Model save/load
- [ ] More optimizers (Adam, RMSprop)
- [ ] Batch training
- [ ] Validation and metrics
- [ ] Convolutional layers

## Comparison

|                  |  Fenix.AI | TensorFlow.js |  Brain.js |
|------------------|-----------|---------------|-----------|
| Size             |     ~10KB |        ~500KB |     ~50KB |
| Dependencies     |         - |     Very many |         - |
| Simplicity       | ★★★★★★ |     ☆☆☆☆★★ | ☆☆★★★★ |
| Functionality    | ☆☆✫★★★ |     ★★★★★★ | ☆☆★★★★ |
| Speed            | ☆☆✫★★★ |     ★★★★★★ | ☆☆★★★★ |

## FAQ

**Q: Works in browser?**  
A: Yes, in any modern browser.

**Q: TypeScript support?**  
A: Not yet, planned.

**Q: GPU training?**  
A: Not yet, WebGL planned.

**Q: Model saving?**  
A: Currently via JSON.stringify(), full implementation planned.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

**You are free to:**
- Use this project for learning and non-commercial purposes
- Modify and adapt the code
- Share your modifications

**You must:**
- Give appropriate credit to Tetra[DEVD]
- Indicate if you made changes
- Use the same license for any derivative works
- **NOT use this project for commercial purposes**

For commercial licensing, please contact megrm.dir@gmail.com.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Support

- [Issues](https://github.com/megrmdir/fenix-ai/issues) for bugs
- Examples in `examples/` folder
- Documentation in code comments

## Acknowledgments

Thanks to all testers and contributors!
