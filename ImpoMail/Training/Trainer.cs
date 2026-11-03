using ImpoMail.Evaluation;
using ImpoMail.Models;
using NetworkModel = ImpoMail.NeuralNetwork.NeuralNetwork;

namespace ImpoMail.Training;

/// <summary>Training hyperparameters and configuration.</summary>
/// <param name="Epochs">Total number of training epochs.</param>
/// <param name="LearningRate">Initial learning rate for gradient descent.</param>
/// <param name="Momentum">Momentum factor (0 = no momentum, 0.9 = standard).</param>
/// <param name="LrDecayFactor">Multiply LR by this factor at each decay step (e.g. 0.5 = halve).</param>
/// <param name="LrDecayEvery">Decay LR every N epochs (0 = no decay).</param>
/// <param name="LogEvery">Print training progress every N epochs.</param>
/// <param name="LogCsvPath">If set, export the training log to this CSV file path.</param>
public sealed record TrainingOptions(
    int Epochs = 2000,
    double LearningRate = 0.05,
    double Momentum = 0.9,
    double LrDecayFactor = 0.5,
    int LrDecayEvery = 500,
    int LogEvery = 100,
    string? LogCsvPath = null);

/// <summary>
/// Trains the neural network using stochastic gradient descent with
/// momentum and learning-rate decay. Logs progress to console and
/// optionally exports a training log CSV.
/// </summary>
public sealed class Trainer
{
    /// <summary>
    /// Runs the full training loop. Each epoch:
    ///   1. Shuffles the training data
    ///   2. Runs forward + backward pass on every sample (backpropagation)
    ///   3. Optionally decays the learning rate
    ///   4. Logs metrics at configured intervals
    /// </summary>
    /// <returns>A <see cref="TrainingLog"/> containing per-epoch metrics.</returns>
    public TrainingLog Train(NetworkModel network, IReadOnlyList<EmailVector> trainingData,
        IReadOnlyList<EmailVector> testData, TrainingOptions options)
    {
        if (trainingData.Count == 0)
            throw new ArgumentException("Training data cannot be empty.", nameof(trainingData));

        var log = new TrainingLog();
        var random = new Random(42);
        var order = Enumerable.Range(0, trainingData.Count).ToArray();
        var currentLr = options.LearningRate;

        for (var epoch = 1; epoch <= options.Epochs; epoch++)
        {
            // ── Learning-rate decay ──
            if (epoch > 1 && options.LrDecayEvery > 0 && epoch % options.LrDecayEvery == 0)
            {
                currentLr *= options.LrDecayFactor;
                Console.WriteLine($"           >> Learning rate decayed to {currentLr:F6}");
            }

            // ── Shuffle training order each epoch (prevents order bias) ──
            Shuffle(order, random);

            // ── Train on every sample (stochastic gradient descent) ──
            var epochLoss = 0d;
            foreach (var index in order)
            {
                var item = trainingData[index];
                epochLoss += network.TrainOne(item.Features, item.Label, currentLr, options.Momentum);
            }

            // ── Log metrics at intervals ──
            if (epoch == 1 || epoch % options.LogEvery == 0 || epoch == options.Epochs)
            {
                var avgLoss = epochLoss / trainingData.Count;
                var metrics = Evaluator.Evaluate(network, testData);
                Console.WriteLine(
                    $"Epoch {epoch,5} | loss {avgLoss:F4} | LR {currentLr:F6} | " +
                    $"test acc {metrics.Accuracy:P1} | F1 {metrics.F1:P1}");

                log.Add(new TrainingLogEntry(epoch, avgLoss, currentLr,
                    metrics.Accuracy, metrics.Precision, metrics.Recall, metrics.F1));
            }
        }

        // ── Export training log CSV if path is configured ──
        if (!string.IsNullOrEmpty(options.LogCsvPath))
        {
            log.ExportCsv(options.LogCsvPath);
            Console.WriteLine($"\nTraining log exported to: {options.LogCsvPath}");
        }

        return log;
    }

    /// <summary>Fisher-Yates shuffle for randomising training order.</summary>
    private static void Shuffle(int[] values, Random random)
    {
        for (var i = values.Length - 1; i > 0; i--)
        {
            var j = random.Next(i + 1);
            (values[i], values[j]) = (values[j], values[i]);
        }
    }
}
