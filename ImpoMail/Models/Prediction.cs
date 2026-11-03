namespace ImpoMail.Models;

public sealed record Prediction(string Importance, double Probability, EmailSample Email);
