namespace ImpoMail.NeuralNetwork;

/// <summary>
/// 3-layer Multi-Layer Perceptron (Input → Hidden → Output) with
/// backpropagation and momentum-based gradient descent.
/// Used for binary classification of email importance.
///
/// Architecture:
///   InputLayer  (17 neurons) — passes features through
///   HiddenLayer (16 neurons) — sigmoid activation, learns patterns
///   OutputLayer (1 neuron)   — sigmoid activation, outputs probability
/// </summary>
public sealed class NeuralNetwork
{
    /// <summary>The input layer (passes features through without transformation).</summary>
    public InputLayer Input { get; }

    /// <summary>The hidden layer with sigmoid activation and learned weights.</summary>
    public HiddenLayer Hidden { get; }

    /// <summary>The output layer producing a single probability value.</summary>
    public OutputLayer Output { get; }

    /// <summary>Number of input features.</summary>
    public int InputSize => Input.Size;

    /// <summary>Number of hidden neurons.</summary>
    public int HiddenSize => Hidden.Size;

    /// <param name="inputSize">Number of features per sample.</param>
    /// <param name="hiddenSize">Number of neurons in the hidden layer.</param>
    /// <param name="seed">Random seed for reproducible weight initialization.</param>
    public NeuralNetwork(int inputSize, int hiddenSize, int seed = 42)
    {
        var random = new Random(seed);
        Input = new InputLayer(inputSize);
        Hidden = new HiddenLayer(inputSize, hiddenSize, random);
        Output = new OutputLayer(hiddenSize, random);
    }

    /// <summary>
    /// Runs the forward pass through all three layers and returns
    /// the output probability (0 = not important, 1 = important).
    /// </summary>
    public double PredictProbability(IReadOnlyList<float> features)
    {
        var inputValues = Input.Forward(features);
        var hiddenActivations = Hidden.Forward(inputValues);
        return Output.Forward(hiddenActivations);
    }

    /// <summary>
    /// Performs one complete training step (forward + backward pass) for a
    /// single sample using backpropagation with momentum.
    ///
    /// Backpropagation steps:
    ///   1. Forward pass — compute activations through all layers
    ///   2. Compute output error — δ_out = prediction − label
    ///   3. Backpropagate to hidden — δ_h = δ_out × w_h × σ'(h)
    ///   4. Update output weights — w -= lr × δ_out × hidden_activation
    ///   5. Update hidden weights — w -= lr × δ_h × input_value
    ///
    /// Returns the binary cross-entropy loss for this sample.
    /// </summary>
    public double TrainOne(IReadOnlyList<float> features, int label,
                           double learningRate, double momentum = 0.9)
    {
        if (label is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(label), "Label must be 0 or 1.");

        // ── Forward pass through all three layers ──
        var inputValues = Input.Forward(features);
        var hiddenActivations = Hidden.Forward(inputValues);
        var prediction = Output.Forward(hiddenActivations);

        // ── Backward pass (backpropagation) ──
        // Step 1: Compute output error (derivative of BCE loss w.r.t. output)
        var outputError = Output.ComputeOutputError(label);

        // Step 2: Propagate error back to hidden layer using chain rule
        var hiddenErrors = Output.ComputeHiddenErrors(outputError, hiddenActivations);

        // Step 3: Update weights with momentum (output layer first, then hidden)
        Output.UpdateWeights(hiddenActivations, outputError, learningRate, momentum);
        Hidden.UpdateWeights(inputValues, hiddenErrors, learningRate, momentum);

        // ── Return binary cross-entropy loss ──
        return -(label * Math.Log(Math.Max(prediction, 1e-12)) +
                 (1 - label) * Math.Log(Math.Max(1 - prediction, 1e-12)));
    }
}
