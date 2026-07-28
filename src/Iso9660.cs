// Minimal ISO9660 reader - just enough for a PS2 game disc image: find the
// Primary Volume Descriptor, walk directory extents, extract files. PS2 DVDs
// are plain ISO9660 with 2048-byte logical sectors and 8.3-ish names carrying
// a ";1" version suffix; no Joliet/Rock Ridge needed. Multi-GB images are fine
// (long offsets throughout).
namespace Quarry;

public sealed class IsoEntry
{
    public required string Name { get; init; }      // "SYSTEM.CNF" (";1" stripped, uppercased)
    public required string Path { get; init; }      // "DATA/SCRIPT/MAIN.SCM"
    public required uint Lba { get; init; }
    public required uint Size { get; init; }
    public required bool IsDirectory { get; init; }
}

public sealed class Iso9660Reader : IDisposable
{
    public const int SectorSize = 2048;

    private readonly FileStream _fs;
    private readonly uint _rootLba, _rootSize;

    public Iso9660Reader(string isoPath)
    {
        _fs = new FileStream(isoPath, FileMode.Open, FileAccess.Read, FileShare.Read,
                             1 << 16, FileOptions.RandomAccess);
        // Volume descriptors start at sector 16; type 1 = PVD, type 255 = terminator.
        Span<byte> sec = stackalloc byte[SectorSize];
        for (int vd = 16; vd < 32; ++vd)
        {
            ReadSector((uint)vd, sec);
            if (sec[1] != (byte)'C' || sec[2] != (byte)'D' || sec[3] != (byte)'0' ||
                sec[4] != (byte)'0' || sec[5] != (byte)'1')
                throw new InvalidDataException("Not an ISO9660 image (no CD001 signature).");
            if (sec[0] == 1)
            {
                // root directory record lives at offset 156 of the PVD
                _rootLba  = BitConverter.ToUInt32(sec.Slice(156 + 2, 4));
                _rootSize = BitConverter.ToUInt32(sec.Slice(156 + 10, 4));
                return;
            }
            if (sec[0] == 255) break;
        }
        throw new InvalidDataException("ISO9660 Primary Volume Descriptor not found.");
    }

    private void ReadSector(uint lba, Span<byte> buf)
    {
        _fs.Position = (long)lba * SectorSize;
        _fs.ReadExactly(buf[..SectorSize]);
    }

    /// Walk one directory extent into entries (self/parent records skipped).
    private List<IsoEntry> ReadDirectory(uint lba, uint size, string dirPath)
    {
        var list = new List<IsoEntry>;
        var buf = new byte[size];
        _fs.Position = (long)lba * SectorSize;
        _fs.ReadExactly(buf);

        int off = 0;
        while (off < buf.Length)
        {
            int len = buf[off];
            if (len == 0)
            {   // zero-length record = rest of this sector is padding; hop to the next sector
                off = (off / SectorSize + 1) * SectorSize;
                continue;
            }
            int nameLen = buf[off + 32];
            if (nameLen == 1 && (buf[off + 33] == 0 || buf[off + 33] == 1))
            { off += len; continue; }                      // "." / ".."

            string rawName = System.Text.Encoding.ASCII.GetString(buf, off + 33, nameLen);
            int semi = rawName.IndexOf(';');
            if (semi >= 0) rawName = rawName[..semi];
            rawName = rawName.TrimEnd('.').ToUpperInvariant;

            bool isDir = (buf[off + 25] & 0x02) != 0;
            list.Add(new IsoEntry
            {
                Name = rawName,
                Path = dirPath.Length == 0 ? rawName : dirPath + "/" + rawName,
                Lba = BitConverter.ToUInt32(buf, off + 2),
                Size = BitConverter.ToUInt32(buf, off + 10),
                IsDirectory = isDir,
            });
            off += len;
        }
        return list;
    }

    /// Full recursive listing (paths use '/' separators, all uppercase).
    public List<IsoEntry> ListAll
    {
        var all = new List<IsoEntry>;
        void Walk(uint lba, uint size, string path)
        {
            foreach (var e in ReadDirectory(lba, size, path))
            {
                all.Add(e);
                if (e.IsDirectory) Walk(e.Lba, e.Size, e.Path);
            }
        }
        Walk(_rootLba, _rootSize, "");
        return all;
    }

    /// Find a single file by exact path ("DATA/TIMECYCP.DAT"), case-insensitive.
    public IsoEntry? Find(string path)
    {
        path = path.Replace('\\', '/').ToUpperInvariant;
        string[] parts = path.Split('/', StringSplitOptions.RemoveEmptyEntries);
        uint lba = _rootLba, size = _rootSize;
        string cur = "";
        for (int i = 0; i < parts.Length; ++i)
        {
            var entries = ReadDirectory(lba, size, cur);
            var hit = entries.FirstOrDefault(e => e.Name == parts[i]);
            if (hit is null) return null;
            if (i == parts.Length - 1) return hit;
            if (!hit.IsDirectory) return null;
            lba = hit.Lba; size = hit.Size; cur = hit.Path;
        }
        return null;
    }

    /// Read a whole (small) file into memory.
    public byte[] ReadAllBytes(IsoEntry e)
    {
        var buf = new byte[e.Size];
        _fs.Position = (long)e.Lba * SectorSize;
        _fs.ReadExactly(buf);
        return buf;
    }

    /// Stream a file out to disk with progress (files up to GBs - gta3.img class).
    public void ExtractTo(IsoEntry e, string destPath, Action<long, long>? progress = null)
    {
        Directory.CreateDirectory(System.IO.Path.GetDirectoryName(destPath)!);
        using var outFs = new FileStream(destPath, FileMode.Create, FileAccess.Write,
                                         FileShare.None, 1 << 20);
        _fs.Position = (long)e.Lba * SectorSize;
        var buf = new byte[1 << 20];
        long left = e.Size, done = 0;
        while (left > 0)
        {
            int n = (int)Math.Min(left, buf.Length);
            _fs.ReadExactly(buf, 0, n);
            outFs.Write(buf, 0, n);
            left -= n; done += n;
            progress?.Invoke(done, e.Size);
        }
    }

    /// A read-only sub-stream over one file inside the image (for the IMG parser --
    /// no need to extract a 900MB archive to walk its directory).
    public Stream OpenRead(IsoEntry e) => new SubStream(_fs, (long)e.Lba * SectorSize, e.Size);

    public void Dispose => _fs.Dispose;
}

/// Read-only window into a larger stream. NOT thread-safe (shares the parent's position).
public sealed class SubStream : Stream
{
    private readonly Stream _base;
    private readonly long _start, _len;
    private long _pos;

    public SubStream(Stream b, long start, long len) { _base = b; _start = start; _len = len; }

    public override bool CanRead => true;
    public override bool CanSeek => true;
    public override bool CanWrite => false;
    public override long Length => _len;
    public override long Position { get => _pos; set => _pos = value; }

    public override int Read(byte[] buffer, int offset, int count)
    {
        long left = _len - _pos;
        if (left <= 0) return 0;
        int n = (int)Math.Min(count, left);
        _base.Position = _start + _pos;
        n = _base.Read(buffer, offset, n);
        _pos += n;
        return n;
    }

    public override long Seek(long offset, SeekOrigin origin)
    {
        _pos = origin switch
        {
            SeekOrigin.Begin => offset,
            SeekOrigin.Current => _pos + offset,
            _ => _len + offset,
        };
        return _pos;
    }

    public override void Flush { }
    public override void SetLength(long value) => throw new NotSupportedException;
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException;
}
