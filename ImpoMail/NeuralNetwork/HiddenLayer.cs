namespace ImpoMail.NeuralNetwork;

/// <summary>
/// Hidden layer of the MLP with sigmoid activation.
/// Connects the input layer to the output layer. Stores weights,
/// biases, and momentum velocities for backpropagation training.
/// </summary>
public sealed class HiddenLayer
{
    /// <summary>Number of neurons feeding into this layer.</summary>
    public int InputSize { get; }

    /// <summary>Number of hidden neurons.</summary>
    public int Size { get; }

    // ── Learnable parameters ──
    internal readonly double[,] Weights;   // [InputSize, Size]
    internal readonly double[] Biases;     // [Size]

    // ── Momentum velocities (used by SGD + momentum optimizer) ──
    private readonly double[,] _weightVelocity;
    private readonly double[] _biasVelocity;

    // ── Forward-pass cache (reused during backpropagation) ──
    internal readonly double[] Activations;

    public HiddenLayer(int inputSize, int size, Random random)
    {
        if (inputSize < 1) throw new ArgumentOutOfRangeException(nameof(inputSize));
        if (size < 1) throw new ArgumentOutOfRangeException(nameof(size));

        InputSize = inputSize;
        Size = size;

        Weights = new double[inputSize, size];
        Biases = new double[size];
        _weightVelocity = new double[inputSize, size];
        _biasVelocity = new double[size];
        Activations = new double[size];

        // He initialization — good starting scale for sigmoid networks
        double scale = Math.Sqrt(2.0 / inputSize);
        for (int i = 0; i < inputSize; i++)
            for (int h = 0; h < size; h++)
                Weights[i, h] = (random.NextDouble() * 2 - 1) * scale;
    }

    /// <summary>
    /// Computes the forward pass: weighted sum + bias → sigmoid activation.
    /// Caches activations for use during backpropagation.
    /// </summary>
    public double[] Forward(double[] inputs)
    {
        for (int h = 0; h < Size; h++)
        {
            double sum = Biases[h];
            for (int i = 0; i < InputSize; i++)
                sum += inputs[i] * Weights[i, h];
            Activations[h] = Sigmoid(sum);
        }
        return Activations;
    }

    /// <summary>
    /// Updates weights and biases using gradient descent with momentum.
    /// velocity = momentum * velocity − learningRate * gradient
    /// weight += velocity
    /// </summary>
    public void UpdateWeights(double[] inputValues, double[] hiddenErrors,
                              double learningRate, double momentum)
    {
        for (int h = 0; h < Size; h++)
        {
            for (int i = 0; i < InputSize; i++)
            {
                double gradient = hiddenErrors[h] * inputValues[i];
                _weightVelocity[i, h] = momentum * _weightVelocity[i, h] - learningRate * gradient;
                Weights[i, h] += _weightVelocity[i, h];
            }
            _biasVelocity[h] = momentum * _biasVelocity[h] - learningRate * hiddenErrors[h];
            Biases[h] += _biasVelocity[h];
        }
    }

    /// <summary>Numerically stable sigmoid activation function.</summary>
    internal static double Sigmoid(double x) =>
        x >= 0 ? 1.0 / (1.0 + Math.Exp(-x)) : Math.Exp(x) / (1.0 + Math.Exp(x));
}
