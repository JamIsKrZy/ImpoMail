namespace ImpoMail.NeuralNetwork;

/// <summary>
/// Input layer of the 3-layer MLP. Accepts raw feature vectors
/// and passes them forward to the hidden layer without transformation.
/// No weights or activation — serves as the entry point of the network.
/// </summary>
public sealed class InputLayer
{
    /// <summary>Number of input neurons (one per feature).</summary>
    public int Size { get; }

    private readonly double[] _values;

    public InputLayer(int size)
    {
        if (size < 1)
            throw new ArgumentOutOfRangeException(nameof(size), "Input layer must have at least 1 neuron.");
        Size = size;
        _values = new double[size];
    }

    /// <summary>
    /// Loads input features into the layer and returns them as doubles
    /// for the hidden layer to consume.
    /// </summary>
    public double[] Forward(IReadOnlyList<float> inputs)
    {
        if (inputs.Count != Size)
            throw new ArgumentException($"Expected {Size} features, got {inputs.Count}.", nameof(inputs));

        for (int i = 0; i < Size; i++)
            _values[i] = inputs[i];

        return _values;
    }
}
