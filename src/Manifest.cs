// Incremental-convert + integrity manifest. Solves the "wait an hour on every
// engine update" problem: each pipeline stage is tagged with a version; the
// manifest records, per stage, that version plus every output file it wrote
// (path + size + sha256). On a re-convert into an existing data/ folder a stage
// is SKIPPED when its version is unchanged AND all its outputs are present with
// matching hashes - so a HUD-only update rebakes only the HUD stage and keeps
// the hour-long world stage verified-and-intact. A standalone verify pass
// checks every recorded file for corruption without rebaking anything.
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Quarry;

public sealed class FileRecord
{
    public string Path { get; set; } = "";   // relative to the data/ dir, '/'-separated
    public long Size { get; set; }
    public string Sha256 { get; set; } = "";
}

public sealed class StageRecord
{
    public int Version { get; set; }
    public List<FileRecord> Outputs { get; set; } = new;
}

public sealed class Manifest
{
    public int SchemaVersion { get; set; } = 1;
    public string ConverterVersion { get; set; } = "";
    public string DiscElf { get; set; } = "";
    public string DiscVer { get; set; } = "";
    public Dictionary<string, StageRecord> Stages { get; set; } = new;

    private const string FileName = "quarry.manifest.json";

    private static readonly JsonSerializerOptions JsonOpts = new
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public static Manifest Load(string dataDir)
    {
        string p = System.IO.Path.Combine(dataDir, FileName);
        if (!File.Exists(p)) return new Manifest;
        try { return JsonSerializer.Deserialize<Manifest>(File.ReadAllText(p)) ?? new Manifest; }
        catch { return new Manifest; }          // corrupt manifest = treat as absent (full rebake)
    }

    public void Save(string dataDir)
    {
        Directory.CreateDirectory(dataDir);
        File.WriteAllText(System.IO.Path.Combine(dataDir, FileName),
                          JsonSerializer.Serialize(this, JsonOpts));
    }

    public static string Sha256(string path)
    {
        using var s = File.OpenRead(path);
        using var h = SHA256.Create;
        return Convert.ToHexString(h.ComputeHash(s)).ToLowerInvariant;
    }

    private static string Rel(string dataDir, string full) =>
        System.IO.Path.GetRelativePath(dataDir, full).Replace('\\', '/');

    /// True if this stage can be skipped: same version AND every recorded output
    /// still present with matching size + hash.
    public bool CanSkip(string stageId, int version, string dataDir, Action<string>? log = null)
    {
        if (!Stages.TryGetValue(stageId, out var st)) return false;
        if (st.Version != version) return false;
        if (st.Outputs.Count == 0) return false;        // never recorded outputs -> rebake
        foreach (var o in st.Outputs)
        {
            string full = System.IO.Path.Combine(dataDir, o.Path.Replace('/', System.IO.Path.DirectorySeparatorChar));
            if (!File.Exists(full)) { log?.Invoke($"   changed: {o.Path} missing"); return false; }
            var fi = new FileInfo(full);
            if (fi.Length != o.Size) { log?.Invoke($"   changed: {o.Path} size"); return false; }
            if (Sha256(full) != o.Sha256) { log?.Invoke($"   changed: {o.Path} hash"); return false; }
        }
        return true;
    }

    /// Snapshot a directory tree as {relpath -> (size, lastWriteUtc ticks)}.
    public static Dictionary<string, (long size, long ticks)> Snapshot(string dir)
    {
        var map = new Dictionary<string, (long, long)>;
        if (!Directory.Exists(dir)) return map;
        foreach (var f in Directory.EnumerateFiles(dir, "*", SearchOption.AllDirectories))
        {
            if (System.IO.Path.GetFileName(f) == FileName) continue;
            var fi = new FileInfo(f);
            map[Rel(dir, f)] = (fi.Length, fi.LastWriteTimeUtc.Ticks);
        }
        return map;
    }

    /// Like Snapshot, but limited to the given subtree roots so parallel stages that
    /// write disjoint folders never cross-attribute each other's files. A root of "" means
    /// the data/ root's top-level files only (non-recursive); any other root is a subdir
    /// walked recursively. The manifest file itself is always excluded.
    public static Dictionary<string, (long size, long ticks)> SnapshotScoped(string dataDir, IEnumerable<string> roots)
    {
        var map = new Dictionary<string, (long, long)>;
        foreach (var root in roots)
        {
            if (root.Length == 0)
            {
                if (!Directory.Exists(dataDir)) continue;
                foreach (var f in Directory.EnumerateFiles(dataDir, "*", SearchOption.TopDirectoryOnly))
                {
                    if (System.IO.Path.GetFileName(f) == FileName) continue;
                    var fi = new FileInfo(f);
                    map[Rel(dataDir, f)] = (fi.Length, fi.LastWriteTimeUtc.Ticks);
                }
            }
            else
            {
                string dir = System.IO.Path.Combine(dataDir, root);
                if (!Directory.Exists(dir)) continue;
                foreach (var f in Directory.EnumerateFiles(dir, "*", SearchOption.AllDirectories))
                {
                    if (System.IO.Path.GetFileName(f) == FileName) continue;
                    var fi = new FileInfo(f);
                    map[Rel(dataDir, f)] = (fi.Length, fi.LastWriteTimeUtc.Ticks);
                }
            }
        }
        return map;
    }

    /// Record a stage's outputs = the files new-or-changed between two snapshots.
    public void RecordStage(string stageId, int version, string dataDir,
                            Dictionary<string, (long size, long ticks)> before,
                            Dictionary<string, (long size, long ticks)> after)
    {
        var st = new StageRecord { Version = version };
        foreach (var (rel, meta) in after)
        {
            if (before.TryGetValue(rel, out var old) && old == meta) continue;  // unchanged
            string full = System.IO.Path.Combine(dataDir, rel.Replace('/', System.IO.Path.DirectorySeparatorChar));
            st.Outputs.Add(new FileRecord { Path = rel, Size = meta.size, Sha256 = Sha256(full) });
        }
        Stages[stageId] = st;
    }

    /// Integrity pass: check every recorded output. Returns (ok, corrupt, missing) counts.
    public (int ok, int bad, int missing) Verify(string dataDir, Action<string> log)
    {
        int ok = 0, bad = 0, missing = 0;
        foreach (var (stageId, st) in Stages)
            foreach (var o in st.Outputs)
            {
                string full = System.IO.Path.Combine(dataDir, o.Path.Replace('/', System.IO.Path.DirectorySeparatorChar));
                if (!File.Exists(full)) { log($"MISSING [{stageId}] {o.Path}"); missing++; }
                else if (new FileInfo(full).Length != o.Size || Sha256(full) != o.Sha256)
                { log($"CORRUPT [{stageId}] {o.Path}"); bad++; }
                else ok++;
            }
        return (ok, bad, missing);
    }
}
