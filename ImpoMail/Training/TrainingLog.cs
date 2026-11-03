using System.Text;

namespace ImpoMail.Training;

/// <summary>
/// Records per-epoch training metrics and exports them to CSV.
/// Deliverable: "A console output or exported CSV file showing
/// the loss decreasing across epochs."
/// </summary>
public sealed class TrainingLog
{
    private readonly List<TrainingLogEntry> _entries = [];

    /// <summary>Adds a new epoch entry to the log.</summary>
    public void Add(TrainingLogEntry entry) => _entries.Add(entry);

    /// <summary>All recorded entries.</summary>
    public IReadOnlyList<TrainingLogEntry> Entries => _entries;

    /// <summary>
    /// Exports the full training log as a CSV file with columns:
    /// Epoch, TrainLoss, LearningRate, TestAccuracy, Precision, Recall, F1.
    /// </summary>
    public void ExportCsv(string filePath)
    {
        var sb = new StringBuilder();
        sb.AppendLine("Epoch,TrainLoss,LearningRate,TestAccuracy,Precision,Recall,F1");
        foreach (var e in _entries)
        {
            sb.AppendLine($"{e.Epoch},{e.TrainLoss:F6},{e.LearningRate:F6}," +
                          $"{e.TestAccuracy:F4},{e.Precision:F4},{e.Recall:F4},{e.F1:F4}");
        }
        File.WriteAllText(filePath, sb.ToString());
    }
}

/// <summary>A single row of the training log.</summary>
public sealed record TrainingLogEntry(
    int Epoch,
    double TrainLoss,
    double LearningRate,
    double TestAccuracy,
    double Precision,
    double Recall,
    double F1);
