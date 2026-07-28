using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
namespace Quarry;

public readonly record struct UpdateResult(bool Reachable, bool UpToDate, string Latest, string? Url);

// Checks the latest GitHub release tag for the converter repo. Offline/error-tolerant:
// a failure returns Reachable=false, never throws to the caller.
public static class UpdateChecker
{
    private const string Api = "https://api.github.com/repos/SandstoneDev/quarry/releases/latest";

    public static async Task<UpdateResult> CheckAsync(string current, HttpClient http, CancellationToken ct)
    {
        try
        {
            if (!http.DefaultRequestHeaders.UserAgent.Any)
                http.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("Quarry", current));
            using var resp = await http.GetAsync(Api, ct);
            if (!resp.IsSuccessStatusCode) return new(false, true, current, null);
            using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync(ct));
            var root = doc.RootElement;
            string latest = root.TryGetProperty("tag_name", out var t) ? (t.GetString ?? current) : current;
            string? url = root.TryGetProperty("html_url", out var u) ? u.GetString : null;
            // "update available" ONLY when the release is NEWER than us - compare the numeric
            // build (v767 vs v766), so being ahead of / equal to the latest release reads as up-to-date.
            int cn = VerNum(current), ln = VerNum(latest);
            bool upToDate = (cn >= 0 && ln >= 0)
                ? cn >= ln
                : string.Equals(latest, current, StringComparison.OrdinalIgnoreCase);
            return new(true, upToDate, latest, url);
        }
        catch { return new(false, true, current, null); }
    }

    // Numeric build out of a tag like "v767" -> 767; -1 if the tag has no digits.
    private static int VerNum(string v)
    {
        var m = System.Text.RegularExpressions.Regex.Match(v ?? "", @"\d+");
        return m.Success && int.TryParse(m.Value, out var n) ? n : -1;
    }
}
