using System.Globalization;
using ImpoMail.Models;

namespace ImpoMail.Data;

/// <summary>
/// Extracts a 17-dimensional feature vector from an email sample.
/// Features include keyword counts (normalized), sender type indicators,
/// text statistics, and a user-profile match score.
///
/// Feature vector layout:
///   [0-7]   Keyword counts per category (normalized 0–1)
///   [8-10]  Sender type one-hot encoding (trusted / friend / unknown)
///   [11-14] Text statistics (length, punctuation, word count)
///   [15]    Keyword density (total hits / word count)
///   [16]    User-profile match score (0–1)
/// </summary>
public sealed class EmailFeatureExtractor
{
    private readonly UserProfile _profile;

    /// <summary>Names of all 17 features, in vector order.</summary>
    public static readonly string[] FeatureNames =
    [
        "work_count",       "family_count",     "bill_count",       "urgent_count",
        "meeting_count",    "deadline_count",   "payment_count",    "question_count",
        "is_trusted",       "is_friend",        "is_unknown",
        "msg_length",       "has_question_mark", "has_exclamation",
        "word_count",       "keyword_density",  "profile_match"
    ];

    // ── Keyword dictionaries (expanded for better coverage) ──

    private static readonly string[] WorkWords =
        ["work", "project", "client", "office", "manager", "report",
         "team", "employee", "colleague", "supervisor", "task", "assignment"];

    private static readonly string[] FamilyWords =
        ["family", "mum", "mom", "dad", "sister", "brother", "home",
         "parent", "children", "child", "school", "relative", "spouse"];

    private static readonly string[] BillWords =
        ["bill", "invoice", "rent", "utility", "overdue",
         "electricity", "water supply", "gas bill", "subscription"];

    private static readonly string[] UrgentWords =
        ["urgent", "immediately", "asap", "important", "action required",
         "critical", "emergency", "alert"];

    private static readonly string[] MeetingWords =
        ["meeting", "schedule", "calendar", "appointment",
         "conference", "interview", "attend"];

    private static readonly string[] DeadlineWords =
        ["deadline", "expires", "tomorrow", "today", "due date",
         "before monday", "end of day", "by friday", "this week"];

    private static readonly string[] PaymentWords =
        ["pay", "paid", "payment", "bank", "salary",
         "transfer", "loan", "repayment", "refund", "transaction"];

    private static readonly string[] QuestionWords =
        ["question", "please reply", "confirm", "response needed",
         "respond", "answer", "verification", "signature required", "approval"];

    private static readonly string[] MedicalWords =
        ["doctor", "medical", "clinic", "hospital", "prescription", "health", "diagnosis"];

    /// <summary>Maps category names to their keyword arrays for profile matching.</summary>
    private static readonly Dictionary<string, string[]> CategoryMap = new()
    {
        ["work"]     = WorkWords,
        ["family"]   = FamilyWords,
        ["bill"]     = BillWords,
        ["urgent"]   = UrgentWords,
        ["meeting"]  = MeetingWords,
        ["deadline"] = DeadlineWords,
        ["payment"]  = PaymentWords,
        ["question"] = QuestionWords,
        ["client"]   = WorkWords,
        ["medical"]  = MedicalWords
    };

    /// <param name="profile">
    /// The user's profile. Determines which keyword categories receive
    /// a higher profile-match score, personalising importance detection.
    /// </param>
    public EmailFeatureExtractor(UserProfile? profile = null)
    {
        _profile = profile ?? UserProfile.General;
    }

    /// <summary>
    /// Transforms an <see cref="EmailSample"/> into a 17-dimensional
    /// feature vector suitable for neural network input.
    /// </summary>
    public EmailVector Transform(EmailSample email)
    {
        var text = $"{email.Subject} {email.Body}".ToLower(CultureInfo.InvariantCulture);
        var words = text.Split([' ', '\t', '\n', '\r'], StringSplitOptions.RemoveEmptyEntries);
        var wordCount = words.Length;

        // ── Keyword counts (how many distinct keywords matched per category) ──
        int workHits     = CountMatches(text, WorkWords);
        int familyHits   = CountMatches(text, FamilyWords);
        int billHits     = CountMatches(text, BillWords);
        int urgentHits   = CountMatches(text, UrgentWords);
        int meetingHits  = CountMatches(text, MeetingWords);
        int deadlineHits = CountMatches(text, DeadlineWords);
        int paymentHits  = CountMatches(text, PaymentWords);
        int questionHits = CountMatches(text, QuestionWords);

        int totalHits = workHits + familyHits + billHits + urgentHits +
                        meetingHits + deadlineHits + paymentHits + questionHits;

        // ── Sender category one-hot encoding ──
        var senderLower = email.SenderCategory.ToLower(CultureInfo.InvariantCulture);
        float isTrusted = senderLower == "trusted" ? 1f : 0f;
        float isFriend  = senderLower == "friend"  ? 1f : 0f;
        float isUnknown = senderLower == "unknown" ? 1f : 0f;

        // ── Text statistics ──
        float msgLength     = Math.Clamp(text.Length / 300f, 0f, 1f);
        float hasQuestion   = text.Contains('?') ? 1f : 0f;
        float hasExclaim    = text.Contains('!') ? 1f : 0f;
        float normWordCount = Math.Clamp(wordCount / 60f, 0f, 1f);
        float keyDensity    = wordCount > 0
            ? Math.Clamp((float)totalHits / wordCount, 0f, 1f)
            : 0f;

        // ── User-profile match score ──
        float profileMatch = ComputeProfileMatch(text);

        // ── Build the 17-element feature vector ──
        var features = new[]
        {
            Normalize(workHits, 4),       // [0]
            Normalize(familyHits, 4),     // [1]
            Normalize(billHits, 4),       // [2]
            Normalize(urgentHits, 3),     // [3]
            Normalize(meetingHits, 3),    // [4]
            Normalize(deadlineHits, 3),   // [5]
            Normalize(paymentHits, 3),    // [6]
            Normalize(questionHits, 3),   // [7]
            isTrusted,                    // [8]
            isFriend,                     // [9]
            isUnknown,                    // [10]
            msgLength,                    // [11]
            hasQuestion,                  // [12]
            hasExclaim,                   // [13]
            normWordCount,                // [14]
            keyDensity,                   // [15]
            profileMatch                  // [16]
        };

        return new EmailVector(features, email.IsImportant ? 1 : 0, email);
    }

    /// <summary>Counts how many distinct keywords from the list appear in the text.</summary>
    private static int CountMatches(string text, string[] keywords)
    {
        int count = 0;
        foreach (var keyword in keywords)
            if (text.Contains(keyword, StringComparison.Ordinal))
                count++;
        return count;
    }

    /// <summary>Normalises a count into 0–1 range with a soft cap.</summary>
    private static float Normalize(int count, int softMax) =>
        Math.Clamp((float)count / softMax, 0f, 1f);

    /// <summary>
    /// Computes a 0–1 score based on how many of the user's priority
    /// categories have keyword matches in the email text.
    /// A higher score means the email matches the user's interests.
    /// </summary>
    private float ComputeProfileMatch(string text)
    {
        if (_profile.PriorityCategories.Length == 0) return 0f;

        int matched = 0;
        foreach (var category in _profile.PriorityCategories)
        {
            if (CategoryMap.TryGetValue(category, out var keywords))
            {
                if (keywords.Any(w => text.Contains(w, StringComparison.Ordinal)))
                    matched++;
            }
        }
        return (float)matched / _profile.PriorityCategories.Length;
    }
}
