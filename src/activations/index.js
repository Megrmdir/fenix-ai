export const activations = {
    // Сигмоида: классика для бинарной классификации
    sigmoid: {
        func: x => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))),
        derivative: x => x * (1 - x)
    },

    // ReLU: стандарт для скрытых слоев
    relu: {
        func: x => Math.max(0, x),
        derivative: x => x > 0 ? 1 : 0
    },

    // Leaky ReLU: решает проблему "мертвых" нейронов
    leakyRelu: {
        func: x => x > 0 ? x : 0.01 * x,
        derivative: x => x > 0 ? 1 : 0.01
    },

    // ELU: Exponential Linear Unit
    elu: {
        func: x => x >= 0 ? x : Math.exp(x) - 1,
        derivative: x => x >= 0 ? 1 : Math.exp(x)
    },

    // Tanh: гиперболический тангенс, [-1, 1]
    tanh: {
        func: x => Math.tanh(x),
        derivative: x => 1 - x * x
    },

    // Swish: x * sigmoid(x)
    swish: {
        func: x => x / (1 + Math.exp(-x)),
        derivative: x => {
            const sig = 1 / (1 + Math.exp(-x));
            return sig + x * sig * (1 - sig);
        }
    },

    // Mish: x * tanh(softplus(x))
    mish: {
        func: x => x * Math.tanh(Math.log(1 + Math.exp(x))),
        derivative: x => {
            const sp = Math.log(1 + Math.exp(x));
            const tsp = Math.tanh(sp);
            const sig = 1 / (1 + Math.exp(-x));
            return tsp + x * sig * (1 - Math.pow(tsp, 2));
        }
    },

    // Sinusoidal: sin(x)
    sin: {
        func: x => Math.sin(x),
        derivative: x => Math.cos(x)
    },

    // Cosine: cos(x)
    cos: {
        func: x => Math.cos(x),
        derivative: x => -Math.sin(x)
    },

    // Gaussian: exp(-x^2)
    gaussian: {
        func: x => Math.exp(-x * x),
        derivative: x => -2 * x * Math.exp(-x * x)
    },

    // Step: дискретная функция
    step: {
        func: x => x > 0 ? 1 : 0,
        derivative: x => 0 // Не используется для обучения, но пусть будет
    },

    // Линейная (identity)
    linear: {
        func: x => x,
        derivative: x => 1
    }
};
