using ImpoMail.Models;

namespace ImpoMail.Data;

/// <summary>
/// Loads and splits the email dataset into training and test sets.
/// Supports user-profile-aware feature extraction.
/// </summary>
public sealed class EmailDataset
{
    /// <summary>Training samples (75% of dataset by default).</summary>
    public IReadOnlyList<EmailVector> Training { get; }

    /// <summary>Test samples (25% of dataset by default).</summary>
    public IReadOnlyList<EmailVector> Test { get; }

    private EmailDataset(IReadOnlyList<EmailVector> training, IReadOnlyList<EmailVector> test)
    {
        Training = training;
        Test = test;
    }

    /// <summary>
    /// Loads the email CSV, transforms each email into a feature vector
    /// using the given user profile, shuffles, and splits into train/test sets.
    /// </summary>
    /// <param name="profile">User profile for personalised feature extraction.</param>
    /// <param name="seed">Random seed for reproducible train/test splits.</param>
    /// <param name="testRatio">Fraction of data reserved for testing (0–1).</param>
    /// <param name="csvPath">Optional explicit path to emails.csv.</param>
    public static EmailDataset Create(UserProfile? profile = null, int seed = 42,
                                      double testRatio = 0.25, string? csvPath = null)
    {
        var extractor = new EmailFeatureExtractor(profile);
        var path = csvPath ?? FindDatasetPath();
        var vectors = CsvEmailLoader.Load(path).Select(extractor.Transform).ToList();
        var random = new Random(seed);
        var shuffled = vectors.OrderBy(_ => random.Next()).ToList();
        var testCount = Math.Max(1, (int)Math.Round(shuffled.Count * testRatio));
        return new EmailDataset(shuffled.Skip(testCount).ToList(), shuffled.Take(testCount).ToList());
    }

    /// <summary>Backward-compatible alias using the General profile.</summary>
    public static EmailDataset CreateDemo(int seed = 42, double testRatio = 0.25, string? csvPath = null)
        => Create(null, seed, testRatio, csvPath);

    private static string FindDatasetPath()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "emails.csv"),
            Path.Combine(AppContext.BaseDirectory, "Data", "emails.csv"),
            Path.Combine(Directory.GetCurrentDirectory(), "Data", "emails.csv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "Data", "emails.csv")
        };

        var path = candidates.Select(Path.GetFullPath).FirstOrDefault(File.Exists);
        return path ?? throw new FileNotFoundException(
            "Could not find emails.csv. Expected it in the application folder or the project Data folder.");
    }
}
