using System.Text;
using ImpoMail.Models;

namespace ImpoMail.Data;

public static class CsvEmailLoader
{
    private static readonly string[] RequiredHeaders = ["Subject", "Body", "SenderCategory", "IsImportant"];

    public static IReadOnlyList<EmailSample> Load(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Email dataset was not found: {filePath}", filePath);

        var lines = File.ReadAllLines(filePath);
        if (lines.Length < 2)
            throw new InvalidDataException("The email CSV must contain a header and at least one data row.");

        var headers = ParseLine(lines[0]);
        if (!headers.SequenceEqual(RequiredHeaders, StringComparer.OrdinalIgnoreCase))
            throw new InvalidDataException($"Expected CSV headers: {string.Join(",", RequiredHeaders)}");

        var emails = new List<EmailSample>();
        for (var lineNumber = 2; lineNumber <= lines.Length; lineNumber++)
        {
            if (string.IsNullOrWhiteSpace(lines[lineNumber - 1])) continue;

            var fields = ParseLine(lines[lineNumber - 1]);
            if (fields.Count != RequiredHeaders.Length)
                throw new InvalidDataException($"Line {lineNumber} must contain exactly {RequiredHeaders.Length} fields.");

            if (!bool.TryParse(fields[3], out var isImportant))
                throw new InvalidDataException($"Line {lineNumber} has an invalid IsImportant value: {fields[3]}");

            emails.Add(new EmailSample(fields[0], fields[1], fields[2], isImportant));
        }

        return emails;
    }

    private static List<string> ParseLine(string line)
    {
        var fields = new List<string>();
        var field = new StringBuilder();
        var insideQuotes = false;

        for (var i = 0; i < line.Length; i++)
        {
            var character = line[i];
            if (character == '"')
            {
                if (insideQuotes && i + 1 < line.Length && line[i + 1] == '"')
                {
                    field.Append('"');
                    i++;
                }
                else
                {
                    insideQuotes = !insideQuotes;
                }
            }
            else if (character == ',' && !insideQuotes)
            {
                fields.Add(field.ToString().Trim());
                field.Clear();
            }
            else
            {
                field.Append(character);
            }
        }

        if (insideQuotes)
            throw new InvalidDataException("CSV contains an unterminated quoted field.");

        fields.Add(field.ToString().Trim());
        return fields;
    }
}
