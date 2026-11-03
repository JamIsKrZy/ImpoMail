using ImpoMail.Models;
using NetworkModel = ImpoMail.NeuralNetwork.NeuralNetwork;

namespace ImpoMail.Evaluation;

public sealed record ClassificationMetrics(int TruePositive, int TrueNegative, int FalsePositive, int FalseNegative)
{
    public int Total => TruePositive + TrueNegative + FalsePositive + FalseNegative;
    public double Accuracy => Total == 0 ? 0 : (double)(TruePositive + TrueNegative) / Total;
    public double Precision => TruePositive + FalsePositive == 0 ? 0 : (double)TruePositive / (TruePositive + FalsePositive);
    public double Recall => TruePositive + FalseNegative == 0 ? 0 : (double)TruePositive / (TruePositive + FalseNegative);
    public double F1 => Precision + Recall == 0 ? 0 : 2 * Precision * Recall / (Precision + Recall);

    public override string ToString() =>
        $"Accuracy: {Accuracy:P1} | Precision: {Precision:P1} | Recall: {Recall:P1} | F1: {F1:P1}";
}

public static class Evaluator
{
    public static ClassificationMetrics Evaluate(NetworkModel network, IEnumerable<EmailVector> data, double threshold = 0.5)
    {
        var tp = 0; var tn = 0; var fp = 0; var fn = 0;
        foreach (var item in data)
        {
            var predicted = network.PredictProbability(item.Features) >= threshold ? 1 : 0;
            if (predicted == 1 && item.Label == 1) tp++;
            else if (predicted == 0 && item.Label == 0) tn++;
            else if (predicted == 1) fp++;
            else fn++;
        }
        return new ClassificationMetrics(tp, tn, fp, fn);
    }

    public static double AverageLoss(NetworkModel network, IEnumerable<EmailVector> data)
    {
        var total = 0d;
        var count = 0;
        foreach (var item in data)
        {
            var probability = Math.Clamp(network.PredictProbability(item.Features), 1e-12, 1 - 1e-12);
            total += -(item.Label * Math.Log(probability) + (1 - item.Label) * Math.Log(1 - probability));
            count++;
        }
        return count == 0 ? 0 : total / count;
    }
}
