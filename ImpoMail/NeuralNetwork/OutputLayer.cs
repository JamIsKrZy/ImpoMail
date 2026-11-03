namespace ImpoMail.NeuralNetwork;

/// <summary>
/// Output layer of the MLP with a single sigmoid neuron for binary classification.
/// Produces a probability (0–1) indicating email importance.
/// </summary>
public sealed class OutputLayer
{
    /// <summary>Number of hidden neurons feeding into this layer.</summary>
    public int InputSize { get; }

    // ── Learnable parameters ──
    internal readonly double[] Weights;  // [InputSize]
    internal double Bias;

    // ── Momentum velocities ──
    private readonly double[] _weightVelocity;
    private double _biasVelocity;

    /// <summary>Last computed output activation (probability).</summary>
    public double Activation { get; private set; }

    public OutputLayer(int inputSize, Random random)
    {
        if (inputSize < 1) throw new ArgumentOutOfRangeException(nameof(inputSize));

        InputSize = inputSize;
        Weights = new double[inputSize];
        _weightVelocity = new double[inputSize];

        // He initialization
        double scale = Math.Sqrt(2.0 / inputSize);
        for (int h = 0; h < inputSize; h++)
            Weights[h] = (random.NextDouble() * 2 - 1) * scale;
    }

    /// <summary>
    /// Forward pass: weighted sum of hidden activations + bias → sigmoid.
    /// </summary>
    public double Forward(double[] hiddenActivations)
    {
        double sum = Bias;
        for (int h = 0; h < InputSize; h++)
            sum += hiddenActivations[h] * Weights[h];
        Activation = HiddenLayer.Sigmoid(sum);
        return Activation;
    }

    /// <summary>
    /// Computes the output error for binary cross-entropy loss:
    /// δ_output = prediction − label
    /// </summary>
    public double ComputeOutputError(int label) => Activation - label;

    /// <summary>
    /// Backpropagates the output error to compute hidden-layer errors
    /// using the chain rule: δ_hidden[h] = δ_output × w[h] × σ'(hidden[h])
    /// where σ'(x) = σ(x) × (1 − σ(x)).
    /// </summary>
    public double[] ComputeHiddenErrors(double outputError, double[] hiddenActivations)
    {
        var errors = new double[InputSize];
        for (int h = 0; h < InputSize; h++)
            errors[h] = outputError * Weights[h] * hiddenActivations[h] * (1 - hiddenActivations[h]);
        return errors;
    }

    /// <summary>
    /// Updates weights and bias using gradient descent with momentum.
    /// </summary>
    public void UpdateWeights(double[] hiddenActivations, double outputError,
                              double learningRate, double momentum)
    {
        for (int h = 0; h < InputSize; h++)
        {
            double gradient = outputError * hiddenActivations[h];
            _weightVelocity[h] = momentum * _weightVelocity[h] - learningRate * gradient;
            Weights[h] += _weightVelocity[h];
        }
        _biasVelocity = momentum * _biasVelocity - learningRate * outputError;
        Bias += _biasVelocity;
    }
}
