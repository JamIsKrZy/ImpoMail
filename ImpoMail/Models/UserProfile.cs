namespace ImpoMail.Models;

/// <summary>
/// Represents the user's context / situation. The neural network uses this
/// to weigh certain email categories higher, personalising importance detection.
/// For example, a "Work" profile prioritises project deadlines and client meetings,
/// while a "Family" profile prioritises bills, payments, and family emergencies.
/// </summary>
public sealed record UserProfile(string Label, string[] PriorityCategories)
{
    /// <summary>Profile for professionals — emphasises work, meetings, deadlines, clients.</summary>
    public static readonly UserProfile Work = new("Work",
        ["work", "meeting", "deadline", "client", "urgent"]);

    /// <summary>Profile for home / family — emphasises family, bills, payments, medical.</summary>
    public static readonly UserProfile Family = new("Family",
        ["family", "bill", "payment", "urgent", "medical"]);

    /// <summary>Profile for students — emphasises deadlines, meetings, questions, payments.</summary>
    public static readonly UserProfile Student = new("Student",
        ["deadline", "meeting", "question", "payment", "urgent"]);

    /// <summary>Default balanced profile — emphasises urgency, deadlines, payments, questions.</summary>
    public static readonly UserProfile General = new("General",
        ["urgent", "deadline", "payment", "question"]);

    /// <summary>Resolves a user-typed label to the matching profile preset.</summary>
    public static UserProfile FromLabel(string label) => label.Trim().ToLowerInvariant() switch
    {
        "work" => Work,
        "family" => Family,
        "student" => Student,
        _ => General
    };

    /// <summary>All available profile labels for the menu.</summary>
    public static readonly string[] AllLabels = ["Work", "Family", "Student", "General"];
}
