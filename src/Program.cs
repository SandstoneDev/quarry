// Quarry entry point. CLI verbs for development smoke tests against reference
// images; no args = the GUI.
// Quarry --probe <iso> print detection result
// Quarry --list <iso> [prefix] list ISO contents (optionally under prefix)
// Quarry --imgdir <iso> <path> list an IMG archive's directory (first 20 + count)
// Quarry --convert <iso> <out> [ids] run the pipeline headless (ids = comma
// list of section ids; default = all available)
namespace Quarry;

public static class Program
{
    [System.Runtime.InteropServices.DllImport("kernel32.dll")]
    private static extern bool AttachConsole(int pid);

    [STAThread]
    public static int Main(string[] args)
    {
        if (args.Length == 0)
        {
            ApplicationConfiguration.Initialize();
            Application.Run(new MainForm());
            return 0;
        }
        AttachConsole(-1);   // WinExe has no console; reattach to the launching terminal for CLI verbs
        try
        {
            switch (args[0])
            {
                case "--probe":
                {
                    using var iso = new Iso9660Reader(args[1]);
                    var info = GameVersion.Probe(iso);
                    Console.WriteLine(info is null ? "not a PS2 disc image" : info.ToString());
                    return info is { Supported: true } ? 0 : 2;
                }
                case "--list":
                {
                    using var iso = new Iso9660Reader(args[1]);
                    string prefix = args.Length > 2 ? args[2].ToUpperInvariant() : "";
                    foreach (var e in iso.ListAll())
                        if (e.Path.StartsWith(prefix))
                            Console.WriteLine($"{(e.IsDirectory ? "D" : " ")} {e.Size,10} {e.Path}");
                    return 0;
                }
                case "--imgdir":
                {
                    using var s = OpenImg(args[1], args.Length > 3 ? args[2] : null);
                    var dir = ImgArchive.ReadDir(s);
                    string? filter = args.Length > 3 ? args[3] : (args.Length > 2 && !args[1].EndsWith(".iso", StringComparison.OrdinalIgnoreCase) ? args[2] : null);
                    foreach (var it in filter is null ? dir.Take(20)
                                     : dir.Where(d => d.Name.Contains(filter, StringComparison.OrdinalIgnoreCase)))
                        Console.WriteLine($"{it.SizeBytes,10} {it.Name}");
                    Console.WriteLine($"... {dir.Count} entries total");
                    return 0;
                }
                case "--imgx":
                {
                    // --imgx <iso> <imgPathOnIso> <entry> <outFile>
                    // --imgx <local.img> <entry> <outFile>
                    bool isIso = args[1].EndsWith(".iso", StringComparison.OrdinalIgnoreCase);
                    using var s = OpenImg(args[1], isIso ? args[2] : null);
                    string entry = isIso ? args[3] : args[2];
                    string outFile = isIso ? args[4] : args[3];
                    var dir = ImgArchive.ReadDir(s);
                    var it = dir.FirstOrDefault(d => d.Name.Equals(entry, StringComparison.OrdinalIgnoreCase))
                             ?? throw new FileNotFoundException(entry);
                    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outFile))!);
                    using var outFs = File.Create(outFile);
                    s.Position = (long)it.OffsetSectors * ImgArchive.SectorSize;
                    var buf = new byte[it.SizeBytes];
                    int read = 0;
                    while (read < buf.Length)
                    {
                        int n = s.Read(buf, read, buf.Length - read);
                        if (n <= 0) break;
                        read += n;
                    }
                    outFs.Write(buf, 0, read);
                    Console.WriteLine($"{it.Name}: {read} bytes -> {outFile}");
                    return 0;
                }
                case "--verify":
                {
                    // Quarry --verify <dataDir>: integrity check against the manifest
                    string dataDir = args[1];
                    var m = Manifest.Load(dataDir);
                    if (m.Stages.Count == 0)
                    {
                        Console.WriteLine("no quarry.manifest.json in " + dataDir +
                                          " (data/ not produced by this converter?)");
                        return 2;
                    }
                    Console.WriteLine($"manifest: converter {m.ConverterVersion}, disc {m.DiscElf} v{m.DiscVer}");
                    var (ok, bad, missing) = m.Verify(dataDir, Console.WriteLine);
                    Console.WriteLine($"verify: {ok} ok, {bad} corrupt, {missing} missing");
                    return (bad == 0 && missing == 0) ? 0 : 1;
                }
                case "--convert":
                {
                    var cx = new ConvertContext
                    {
                        IsoPath = args[1],
                        OutDir = args[2],
                        TempDir = Path.Combine(Path.GetTempPath(), "quarry_cli"),
                        Log = Console.WriteLine,
                    };
                    // optional 3rd arg = comma list of section ids; default = all available
                    var ids = args.Length > 3
                        ? args[3].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToHashSet()
                        : ConvertPipeline.Sections.Where(s => s.Available).Select(s => s.Id).ToHashSet();
                    ids.Add("core");                 // core always runs
                    bool ok;
                    try { ok = ConvertPipeline.Run(cx, ids); } finally { cx.Iso?.Dispose(); }
                    return ok ? 0 : 1;
                }
                default:
                    Console.WriteLine("usage: Quarry [--probe|--list|--imgdir|--imgx|--convert|--verify] ...");
                    return 64;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("error: " + ex.Message);
            return 1;
        }
    }

    /// Open an IMG archive stream: from inside an ISO (imgPathOnIso given) or a
    /// local.img file. Caller disposes.
    private static Stream OpenImg(string path, string? imgPathOnIso)
    {
        if (imgPathOnIso is not null)
        {
            var iso = new Iso9660Reader(path);      // kept alive by the SubStream's parent FileStream lifetime below
            var e = iso.Find(imgPathOnIso) ?? throw new FileNotFoundException(imgPathOnIso);
            return iso.OpenRead(e);                  // NOTE: leaks the reader for CLI one-shots; fine for a tool process
        }
        return new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16);
    }
}
