using ImpoMail.Data;
using ImpoMail.Evaluation;
using ImpoMail.Models;
using ImpoMail.NeuralNetwork;
using ImpoMail.Training;

// ══════════════════════════════════════════════════════════════
//  ImpoMail — Smart Email Importance Classifier
//  Binary classification using a 3-layer MLP with backpropagation
// ══════════════════════════════════════════════════════════════

Console.WriteLine("==================================================");
Console.WriteLine("  ImpoMail - Smart Email Importance Classifier");
Console.WriteLine("  Binary Classification with Neural Network (MLP)");
Console.WriteLine("==================================================\n");

// ── Step 1: Select user profile ──────────────────────────────
Console.WriteLine("Select your profile (determines importance priorities):");
for (int i = 0; i < UserProfile.AllLabels.Length; i++)
    Console.WriteLine($"  {i + 1}. {UserProfile.AllLabels[i]}");

Console.Write("\nChoice [1-4, default=4]: ");
var profileInput = Console.ReadLine()?.Trim();
int profileChoice = int.TryParse(profileInput, out var pc) && pc >= 1 && pc <= 4 ? pc : 4;
var profile = UserProfile.FromLabel(UserProfile.AllLabels[profileChoice - 1]);
Console.WriteLine($"-> Using profile: {profile.Label}");
Console.WriteLine($"   Priority categories: {string.Join(", ", profile.PriorityCategories)}\n");

// ── Step 2: Load and prepare dataset ─────────────────────────
var dataset = EmailDataset.Create(profile);

// Count class distribution
int trainPos = dataset.Training.Count(v => v.Label == 1);
int trainNeg = dataset.Training.Count(v => v.Label == 0);
int testPos  = dataset.Test.Count(v => v.Label == 1);
int testNeg  = dataset.Test.Count(v => v.Label == 0);

Console.WriteLine("Dataset Summary:");
Console.WriteLine($"  Training: {dataset.Training.Count} samples ({trainPos} important, {trainNeg} not important)");
Console.WriteLine($"  Test:     {dataset.Test.Count} samples ({testPos} important, {testNeg} not important)");
Console.WriteLine($"  Features: {EmailFeatureExtractor.FeatureNames.Length}");
Console.WriteLine($"  Feature names: {string.Join(", ", EmailFeatureExtractor.FeatureNames)}\n");

// ── Step 3: Create and train neural network ──────────────────
var network = new NeuralNetwork(
    inputSize: EmailFeatureExtractor.FeatureNames.Length,  // 17 features
    hiddenSize: 16,                                        // 16 hidden neurons
    seed: 42);

Console.WriteLine("Neural Network Architecture:");
Console.WriteLine($"  Input Layer:  {network.InputSize} neurons (one per feature)");
Console.WriteLine($"  Hidden Layer: {network.HiddenSize} neurons (sigmoid activation)");
Console.WriteLine($"  Output Layer: 1 neuron (sigmoid activation)");
Console.WriteLine("  Loss: Binary Cross-Entropy");
Console.WriteLine("  Optimizer: SGD with Momentum (0.9)");
Console.WriteLine("  Learning Rate Decay: halve every 500 epochs\n");

var trainer = new Trainer();
var logPath = Path.Combine(AppContext.BaseDirectory, "training_log.csv");

Console.WriteLine("Training Log:");
Console.WriteLine(new string('-', 85));

var log = trainer.Train(
    network,
    dataset.Training,
    dataset.Test,
    new TrainingOptions(
        Epochs: 2000,
        LearningRate: 0.05,
        Momentum: 0.9,
        LrDecayFactor: 0.5,
        LrDecayEvery: 500,
        LogEvery: 100,
        LogCsvPath: logPath));

Console.WriteLine(new string('-', 85));

// ── Step 4: Final evaluation ─────────────────────────────────
var metrics = Evaluator.Evaluate(network, dataset.Test);

Console.WriteLine("\n=== Final Test Results ===");
Console.WriteLine($"  {metrics}");
Console.WriteLine();
Console.WriteLine("Confusion Matrix:");
Console.WriteLine("                   Predicted");
Console.WriteLine("                Important  Not Imp.");
Console.WriteLine($"  Actual Imp.     {metrics.TruePositive,4}       {metrics.FalseNegative,4}");
Console.WriteLine($"  Actual Not      {metrics.FalsePositive,4}       {metrics.TrueNegative,4}");

// ── Step 5: Demo predictions ─────────────────────────────────
var extractor = new EmailFeatureExtractor(profile);
var examples = new[]
{
    new EmailSample("Urgent client meeting",
        "The project deadline is tomorrow. Please reply.", "trusted", true),
    new EmailSample("Weekend promotion",
        "Enjoy our latest shopping offers and deals.", "unknown", false),
    new EmailSample("Rent invoice due",
        "Your rent payment is due tomorrow. Please make the transfer.", "unknown", true),
    new EmailSample("New podcast episode",
        "Listen to this week's top podcast picks.", "friend", false),
    new EmailSample("Family doctor appointment",
        "Mom's doctor appointment is scheduled for today. Please confirm.", "trusted", true),
};

Console.WriteLine("\n-- Sample Predictions --");
foreach (var email in examples)
{
    var vector = extractor.Transform(email);
    var probability = network.PredictProbability(vector.Features);
    var label = probability >= 0.5 ? "IMPORTANT" : "NOT IMPORTANT";
    var correct = (probability >= 0.5) == email.IsImportant ? "[correct]" : "[wrong]";
    Console.WriteLine($"  {label,-14} ({probability:P1}) | [{email.SenderCategory}] {email.Subject} {correct}");
}

// ── Step 6: Interactive mode ─────────────────────────────────
Console.WriteLine("\n==========================================");
Console.WriteLine("  Interactive Demo (type 'quit' to exit)");
Console.WriteLine("==========================================\n");

while (true)
{
    Console.Write("Subject (or 'quit'): ");
    var subject = Console.ReadLine()?.Trim();
    if (string.IsNullOrEmpty(subject) || subject.Equals("quit", StringComparison.OrdinalIgnoreCase))
        break;

    Console.Write("Body: ");
    var body = Console.ReadLine()?.Trim() ?? "";

    Console.Write("Sender [trusted/friend/unknown]: ");
    var sender = Console.ReadLine()?.Trim() ?? "unknown";
    if (sender != "trusted" && sender != "friend" && sender != "unknown")
        sender = "unknown";

    var testEmail = new EmailSample(subject, body, sender, false);
    var testVector = extractor.Transform(testEmail);
    var prob = network.PredictProbability(testVector.Features);
    var result = prob >= 0.5 ? "IMPORTANT" : "NOT IMPORTANT";

    Console.WriteLine($"\n  -> {result} ({prob:P1} confidence)\n");
}

Console.WriteLine("Goodbye!");
