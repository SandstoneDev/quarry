// IMG archive (VER2, the format all this game generation's archives use):
// header: "VER2" magic, u32 entry count
// entry[]: u32 offset (2048-byte sectors), u16 streamingSize (sectors),
// u16 sizeInArchive (0 on retail), char name[24] (NUL-padded)
// The parser reads from any Stream (a SubStream over the ISO works - no need
// to extract a 900MB archive before walking it).
namespace Quarry;

public sealed record ImgEntry(string Name, uint OffsetSectors, uint SizeBytes);

public static class ImgArchive
{
    public const int SectorSize = 2048;

    public static List<ImgEntry> ReadDir(Stream s)
    {
        using var br = new BinaryReader(s, System.Text.Encoding.ASCII, leaveOpen: true);
        s.Position = 0;
        if (br.ReadByte() != 'V' || br.ReadByte() != 'E' || br.ReadByte() != 'R' || br.ReadByte() != '2')
            throw new InvalidDataException("Not a VER2 IMG archive.");
        uint count = br.ReadUInt32();
        var list = new List<ImgEntry>((int)count);
        for (uint i = 0; i < count; ++i)
        {
            uint off = br.ReadUInt32();
            ushort streamingSectors = br.ReadUInt16();
            ushort sizeInArchive = br.ReadUInt16();          // 0 on retail discs
            byte[] nameBytes = br.ReadBytes(24);
            int z = Array.IndexOf(nameBytes, (byte)0); if (z < 0) z = 24;
            string name = System.Text.Encoding.ASCII.GetString(nameBytes, 0, z);
            uint sectors = sizeInArchive != 0 ? sizeInArchive : streamingSectors;
            list.Add(new ImgEntry(name, off, sectors * (uint)SectorSize));
        }
        return list;
    }

    /// Extract entries matching `filter` (null = all) into outDir.
    public static int Extract(Stream img, IEnumerable<ImgEntry> entries, string outDir,
                              Func<ImgEntry, bool>? filter = null,
                              Action<string, int, int>? progress = null)
    {
        Directory.CreateDirectory(outDir);
        var picked = (filter is null ? entries : entries.Where(filter)).ToList();
        var buf = new byte[1 << 20];
        int done = 0;
        foreach (var e in picked)
        {
            string dest = Path.Combine(outDir, e.Name);
            using var outFs = new FileStream(dest, FileMode.Create, FileAccess.Write,
                                             FileShare.None, 1 << 16);
            img.Position = (long)e.OffsetSectors * SectorSize;
            long left = e.SizeBytes;
            while (left > 0)
            {
                int n = img.Read(buf, 0, (int)Math.Min(left, buf.Length));
                if (n <= 0) throw new EndOfStreamException($"IMG short read at '{e.Name}'.");
                outFs.Write(buf, 0, n);
                left -= n;
            }
            ++done;
            progress?.Invoke(e.Name, done, picked.Count);
        }
        return done;
    }
}
