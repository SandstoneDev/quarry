// Disc identification. Both reference PS2 discs boot the SAME elf id
// (SLES_525.41) - the game revision is the "VER = x.yy" line in SYSTEM.CNF
// (verified 2026-07-23 against the v1.03 and v2.01 reference images), so
// detection is: parse SYSTEM.CNF, match the elf id table, read VER.
namespace Quarry;

/// Converter build identity, stamped into every data/ manifest.
public static class QuarryInfo
{
    public const string Version = "v1009";  // release tag; bump per release (see RELEASE_PLAYBOOK). UpdateChecker string-compares this to the latest GitHub release tag.
                                            // ★ The converter version TRACKS THE ENGINE BUILD NUMBER. Quarry sat at
                                            // v915 while the engine reached b1005, so a user could not tell which
                                            // converter matches the build they are running - and that pairing is
                                            // exactly what matters, since the engine reads what this tool bakes.
                                            // One number for both from here on.
}

public sealed record DiscInfo(string ElfId, string Ver, string VMode, bool Supported)
{
    public override string ToString() =>
        $"{ElfId} v{Ver} ({VMode}){(Supported ? "" : " - UNSUPPORTED")}";
}

public static class GameVersion
{
    // elf ids this converter understands (PAL Europe/Australia five-language disc).
    // Extend here when more dumps are verified.
    private static readonly HashSet<string> Supported = new()
    {
        "SLES_525.41",
    };

    /// Parse SYSTEM.CNF text (tiny: BOOT2/VER/VMODE lines).
    public static DiscInfo Parse(string systemCnf)
    {
        string elf = "", ver = "?", vmode = "?";
        foreach (var raw in systemCnf.Split('\n'))
        {
            var line = raw.Trim();
            int eq = line.IndexOf('=');
            if (eq < 0) continue;
            string key = line[..eq].Trim().ToUpperInvariant();
            string val = line[(eq + 1)..].Trim();
            switch (key)
            {
                case "BOOT2":
                    // "cdrom0:\SLES_525.41;1" -> "SLES_525.41"
                    elf = val;
                    int bs = elf.LastIndexOf('\\'); if (bs >= 0) elf = elf[(bs + 1)..];
                    int semi = elf.IndexOf(';'); if (semi >= 0) elf = elf[..semi];
                    elf = elf.Trim();
                    break;
                case "VER":   ver = val; break;
                case "VMODE": vmode = val; break;
            }
        }
        return new DiscInfo(elf, ver, vmode, Supported.Contains(elf));
    }

    /// Probe an ISO image: read SYSTEM.CNF from the root. Null = not a PS2 disc image.
    public static DiscInfo? Probe(Iso9660Reader iso)
    {
        var cnf = iso.Find("SYSTEM.CNF");
        if (cnf is null) return null;
        return Parse(System.Text.Encoding.ASCII.GetString(iso.ReadAllBytes(cnf)));
    }
}



