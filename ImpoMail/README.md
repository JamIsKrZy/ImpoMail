# ImpoMail

A .NET 10 console application that classifies emails as **important** or **not important** using a custom three-layer neural network:

`input features -> hidden layer -> output probability`

No machine-learning or third-party packages are used.

## Files

- `Program.cs` - runs training, evaluation, and example predictions.
- `Models/EmailSample.cs` - labeled email data model and numeric vector model.
- `Models/Prediction.cs` - prediction model for future UI/API use.
- `Data/EmailFeatureExtractor.cs` - converts email text and sender context into 10 numeric inputs.
- `Data/EmailDataset.cs` - loads the CSV and performs the deterministic train/test split.
- `Data/CsvEmailLoader.cs` - parses and validates the external CSV file.
- `Data/emails.csv` - all labeled email examples.
- `NeuralNetwork/NeuralNetwork.cs` - input-hidden-output network and backpropagation.
- `Training/Trainer.cs` - shuffled training loop and loss logging.
- `Evaluation/Metrics.cs` - loss, accuracy, precision, recall, and F1 calculation.

## Run

Run the project from Visual Studio or with:

```powershell
dotnet run --project ImpoMail\ImpoMail.csproj
```

The output includes a loss log, test metrics, a confusion matrix, and two example predictions.

## Fine-tuning guide

Change one setting at a time in `Program.cs` and record the result:

```csharp
new TrainingOptions(Epochs: 1000, LearningRate: 0.08, LogEvery: 100)
```

The network size is configured here:

```csharp
new NeuralNetwork(EmailFeatureExtractor.FeatureNames.Length, hiddenSize: 8, seed: 42)
```

Recommended experiments:

1. **Learning rate:** test `0.01`, `0.03`, `0.08`, and `0.15`. A rate that is too high makes loss unstable; a rate that is too low learns slowly.
2. **Hidden neurons:** test `4`, `8`, and `16`. More neurons can learn more patterns but can overfit a small dataset.
3. **Epochs:** test `500`, `1000`, and `2000`. Stop increasing epochs when test F1 stops improving.
4. **Decision threshold:** the current threshold is `0.5`. If missing important emails is more serious than showing extra emails, lower it, for example to `0.4`, and compare recall.
5. **Features:** add words that are meaningful for the user's work, family, and bill categories. Keep features based on information available before classification.
6. **Random seed:** keep `42` while comparing experiments so changes are reproducible.

Use **test F1**, recall, and the confusion matrix as the primary measures. Accuracy alone can look good when one class is more common than the other.

## Replacing the demo data

The included data is intentionally small and hand-labeled so the application runs immediately. It is not enough for a real email filter. Add or edit rows in `Data/emails.csv` using this format:

```text
Subject,Body,SenderCategory,IsImportant
Client deadline,The report is due tomorrow, trusted, true
Newsletter,Read our weekly news, unknown, false
```

The CSV is copied to the build output automatically. Do not add email records back into `EmailDataset.cs`; edit `Data/emails.csv` instead. Quote fields when they contain commas.

Recommended dataset process:

1. Collect and anonymize at least several hundred emails.
2. Define the label policy before labeling: for example, important means the user must read or act on it.
3. Label examples consistently as important or not important.
4. Keep related emails from the same conversation in the same split to avoid leakage.
5. Reserve the test set until all hyperparameter tuning is finished.
6. Compare the final model against a simple keyword baseline.

## Activity report points

- **Preprocessing:** lower-case text, keyword indicators, sender category indicator, and normalized message length.
- **Training:** stochastic gradient descent with sigmoid activation and backpropagation.
- **Loss:** binary cross-entropy, printed every 100 epochs.
- **Evaluation:** accuracy, precision, recall, F1, and confusion matrix.
- **Limitations:** keyword features do not understand language context, sarcasm, or unseen vocabulary. A larger labeled dataset and stronger text representation would be needed for production use.
