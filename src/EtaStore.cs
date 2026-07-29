using System.Text.Json;
namespace Quarry;

// Per-section time estimates. Ships hand-set built-in defaults; learns the user's
// real per-section durations (moving average of the last few runs) into a JSON
// cache under the given base dir (MainForm passes %LOCALAPPDATA%/Quarry).
public sealed class EtaStore
{
    private const int Window = 5;
    private readonly string _path;
    private Dictionary<string, List<double>> _samples;   // sectionId -> recent seconds

    private static readonly Dictionary<string, int> BuiltinSeconds = new()
    {
        ["core"] = 5, ["world"] = 6000, ["timecyc"] = 3, ["zones"] = 3,   // world ~100 min (day+night decode + col/tex/lod + lz4)
        ["audio"] = 90, ["vehicles"] = 180, ["cutscenes"] = 60,
        ["interiors"] = 45, ["hud"] = 30, ["peds"] = 120,
    };

    public static TimeSpan Builtin(string id) =>
        TimeSpan.FromSeconds(BuiltinSeconds.TryGetValue(id, out var s) ? s : 30);

    public EtaStore(string baseDir)
    {
        Directory.CreateDirectory(baseDir);
        _path = Path.Combine(baseDir, "eta.json");
        _samples = Load();
    }

    private Dictionary<string, List<double>> Load()
    {
        try
        {
            if (File.Exists(_path))
                return JsonSerializer.Deserialize<Dictionary<string, List<double>>>(File.ReadAllText(_path))
                       ?? new();
        }
        catch { /* corrupt cache -> start fresh */ }
        return new();
    }

    public TimeSpan Estimate(string sectionId)
    {
        if (_samples.TryGetValue(sectionId, out var xs) && xs.Count > 0)
            return TimeSpan.FromSeconds(xs.Average());
        return Builtin(sectionId);
    }

    public void Record(string sectionId, TimeSpan elapsed)
    {
        if (!_samples.TryGetValue(sectionId, out var xs)) { xs = new(); _samples[sectionId] = xs; }
        xs.Add(elapsed.TotalSeconds);
        while (xs.Count > Window) xs.RemoveAt(0);
        try { File.WriteAllText(_path, JsonSerializer.Serialize(_samples)); } catch { /* best effort */ }
    }
}
