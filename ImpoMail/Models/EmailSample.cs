namespace ImpoMail.Models;

public sealed record EmailSample(
    string Subject,
    string Body,
    string SenderCategory,
    bool IsImportant);

public sealed record EmailVector(float[] Features, int Label, EmailSample Source);
