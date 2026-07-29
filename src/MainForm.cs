// Quarry GUI: pick a disc image -> auto-detect -> pick an output folder ->
// analyze what's already built -> Convert with per-section live status + ETA.
// Deliberately plain WinForms (one exe, no deps).
using System.Threading;

namespace Quarry;

public sealed class MainForm : Form
{
    private readonly TextBox _isoBox = new() { Anchor = AnchorStyles.Left | AnchorStyles.Top | AnchorStyles.Right };
    private readonly TextBox _outBox = new() { Anchor = AnchorStyles.Left | AnchorStyles.Top | AnchorStyles.Right };
    private readonly Label _detect = new() { AutoSize = true, Text = "no image selected" };
    private readonly Button _convert = new() { Text = "Convert", Enabled = false };
    private readonly Button _cancel = new() { Text = "Cancel", Enabled = false };
    // A convert that goes wrong is diagnosed from the log file, and a tester has to be able
    // to FIND it. The path is printed when the run ends, but that is the one moment nobody is
    // reading, and a failed run is exactly when it matters. This opens the folder outright.
    private readonly Button _openLog = new() { Text = "Open log folder" };
    private readonly Label _eta = new() { AutoSize = true, Text = "" };
    private readonly TextBox _log = new()
    {
        Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Vertical,
        Anchor = AnchorStyles.Left | AnchorStyles.Top | AnchorStyles.Right | AnchorStyles.Bottom,
        Font = new Font(FontFamily.GenericMonospace, 8.5f),
    };
    private readonly Panel _sections = new()
    {
        AutoScroll = true, BorderStyle = BorderStyle.FixedSingle,
        Anchor = AnchorStyles.Left | AnchorStyles.Top | AnchorStyles.Right,
    };
    private readonly List<SectionRow> _rows = new();   // one per ConvertPipeline.Section, in order
    private readonly Label _version = new() { AutoSize = true, ForeColor = Color.Gray, Anchor = AnchorStyles.Left | AnchorStyles.Bottom };
    private readonly Label _updStatus = new() { AutoSize = true, Text = "", ForeColor = Color.Gray, Anchor = AnchorStyles.Left | AnchorStyles.Bottom };
    private readonly Button _checkUpd = new() { Text = "Check for updates", Anchor = AnchorStyles.Right | AnchorStyles.Bottom };
    private readonly ToolTip _tips = new() { AutoPopDelay = 15000 };

    private CancellationTokenSource? _cts;
    private readonly EtaStore _eta_store = new(Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Quarry"));
    private readonly System.Net.Http.HttpClient _http = new();
    private System.IO.StreamWriter? _logFile;   // timestamped convert log written alongside the UI log
    private readonly object _logLock = new();

    public MainForm()
    {
        Text = "Quarry - game data converter for Sandstone Engine";
        MinimumSize = new Size(580, 720);
        Size = new Size(660, 820);
        _version.Text = QuarryInfo.Version;

        var isoBtn = new Button { Text = "...", Anchor = AnchorStyles.Top | AnchorStyles.Right };
        var outBtn = new Button { Text = "...", Anchor = AnchorStyles.Top | AnchorStyles.Right };
        isoBtn.Click += (_, _) => PickIso();
        outBtn.Click += (_, _) => PickOut();
        _convert.Click += (_, _) => RunConvert();
        _cancel.Click += (_, _) => _cts?.Cancel();
        _openLog.Click += (_, _) => OpenLogFolder();
        _checkUpd.Click += async (_, _) => await CheckUpdates(manual: true);

        var isoLbl = new Label { Text = "Disc image (.iso) of YOUR OWN game copy:", AutoSize = true };
        var outLbl = new Label { Text = "Output data folder (goes next to the engine EBOOT):", AutoSize = true };

        SuspendLayout();
        int x = 12, w = ClientSize.Width - 24;
        isoLbl.SetBounds(x, 12, w, 18);
        _isoBox.SetBounds(x, 32, w - 40, 24);
        isoBtn.SetBounds(x + w - 34, 32, 34, 24);
        _detect.SetBounds(x, 60, w, 18);
        outLbl.SetBounds(x, 84, w, 18);
        _outBox.SetBounds(x, 104, w - 40, 24);
        outBtn.SetBounds(x + w - 34, 104, 34, 24);

        // per-section rows: build in order, add to the panel in REVERSE (Dock=Top stacks
        // the last-added on top, so reversing yields top-to-bottom Sections order).
        foreach (var sec in ConvertPipeline.Sections)
        {
            var row = new SectionRow(sec.Id, sec.Name, sec.DefaultOn);
            if (sec.Id == "core") { row.Checked = true; row.CheckEnabled = false; }   // always on
            _tips.SetToolTip(row, sec.Desc);
            _rows.Add(row);
        }
        for (int i = _rows.Count - 1; i >= 0; i--) _sections.Controls.Add(_rows[i]);
        int secH = ConvertPipeline.Sections.Length * 30 + 6;   // 30px per row -> all rows fit, no scroll
        _sections.SetBounds(x, 134, w, secH);

        int afterSections = 134 + secH + 10;
        _convert.SetBounds(x, afterSections, 120, 30);
        _cancel.SetBounds(x + 130, afterSections, 90, 30);
        _openLog.SetBounds(x + 230, afterSections, 130, 30);
        _eta.SetBounds(x + 372, afterSections + 7, w - 372, 18);

        int logTop = afterSections + 42;
        _log.SetBounds(x, logTop, w, ClientSize.Height - logTop - 42);

        int stripY = ClientSize.Height - 30;
        _version.SetBounds(x, stripY + 4, 44, 18);
        _updStatus.SetBounds(x + 52, stripY + 4, w - 52 - 175, 18);
        _checkUpd.SetBounds(x + w - 165, stripY, 165, 26);

        Controls.AddRange(new Control[]
        {
            isoLbl, _isoBox, isoBtn, _detect, outLbl, _outBox, outBtn,
            _sections, _convert, _cancel, _openLog, _eta, _log, _version, _updStatus, _checkUpd,
        });
        ResumeLayout();

        _ = CheckUpdates(manual: false);   // quiet auto-check on launch (offline-tolerant, no popup)
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing) { CloseLogFile(); _cts?.Dispose(); _http.Dispose(); _tips.Dispose(); }
        base.Dispose(disposing);
    }

    private void PickIso()
    {
        using var dlg = new OpenFileDialog { Filter = "Disc images (*.iso)|*.iso|All files|*.*" };
        if (dlg.ShowDialog(this) != DialogResult.OK) return;
        _isoBox.Text = dlg.FileName;
        try
        {
            using var iso = new Iso9660Reader(dlg.FileName);
            var info = GameVersion.Probe(iso);
            _detect.Text = info is null ? "not a PS2 disc image" : $"detected: {info}";
            _detect.ForeColor = info is { Supported: true } ? Color.DarkGreen : Color.Firebrick;
            _convert.Enabled = info is { Supported: true } && _outBox.Text.Length > 0;
        }
        catch (Exception ex)
        {
            _detect.Text = "error: " + ex.Message;
            _detect.ForeColor = Color.Firebrick;
            _convert.Enabled = false;
        }
    }

    private void PickOut()
    {
        using var dlg = new FolderBrowserDialog { Description = "Where to build the data folder" };
        if (dlg.ShowDialog(this) != DialogResult.OK) return;
        _outBox.Text = Path.Combine(dlg.SelectedPath, "data");
        _convert.Enabled = _detect.ForeColor == Color.DarkGreen;
        AnalyzeFolder();
    }

    // Background: read the manifest in the chosen data/ folder and paint each row's
    // status badge + time estimate, and a header summary. Non-fatal (DataAnalyzer degrades).
    private void AnalyzeFolder()
    {
        string dir = _outBox.Text;
        if (dir.Length == 0) return;
        _ = Task.Run(() =>
        {
            try
            {
                int need = 0; TimeSpan total = TimeSpan.Zero;
                foreach (var sec in ConvertPipeline.Sections)
                {
                    var a = DataAnalyzer.AnalyzeSection(sec, dir, _eta_store);
                    var row = _rows.First(r => r.SectionId == sec.Id);
                    row.SetStatus(a.Status); row.SetEstimate(a.Estimate);
                    if (a.Status != SectionStatus.UpToDate) { need++; total += a.Estimate; }
                }
                int n = need; var t = total;
                if (IsHandleCreated)
                    BeginInvoke(() => _eta.Text = n == 0
                        ? "all sections up to date"
                        : $"{n} section(s) to build (about {Math.Round(t.TotalMinutes)} min)");
            }
            catch { /* analysis is best-effort; a form close or IO hiccup must not crash the app */ }
        });
    }

    private async void RunConvert()
    {
        _convert.Enabled = false; _cancel.Enabled = true; _log.Clear();
        _cts?.Dispose();
        _cts = new CancellationTokenSource();
        string tempDir = Path.Combine(Path.GetTempPath(), "quarry_" + Environment.TickCount64);
        var enabled = ConvertPipeline.Sections
            .Where(s => s.Id == "core" || _rows.First(r => r.SectionId == s.Id).Checked).ToList();

        string logPath = OpenLogFile();   // timestamped file log so real per-step timings survive the run
        AppendLog($"log file: {logPath}");   // stated up front: a run that hangs never reaches the closing line

        var cx = new ConvertContext
        {
            IsoPath = _isoBox.Text, OutDir = _outBox.Text, TempDir = tempDir, Ct = _cts.Token,
            Log = AppendLog,
        };
        var progress = new Progress<SectionEvent>(e =>
        {
            var row = _rows.FirstOrDefault(r => r.SectionId == e.SectionId);
            if (row is not null) { row.SetStatus(e.State); if (e.Percent > 0) row.SetProgress(e.Percent); }
            if (e.Line is not null) AppendLog($"[{e.SectionId}] {e.Line}");
        });
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var timer = new System.Windows.Forms.Timer { Interval = 1000 };
        timer.Tick += (_, _) => _eta.Text = $"Elapsed {sw.Elapsed:hh\\:mm\\:ss}";   // hh: shows hours (mm:ss alone hid them)
        timer.Start();
        bool ok;
        try { ok = await SectionRunner.RunAsync(cx, enabled, progress, _eta_store, _cts.Token); }
        catch (OperationCanceledException) { ok = false; AppendLog("== CANCELLED =="); }
        finally { cx.Iso?.Dispose(); timer.Stop(); }
        try { Directory.Delete(tempDir, recursive: true); } catch { /* best-effort */ }
        AppendLog(ok ? "== SUCCESS ==" : "== DONE (with issues) ==");
        AppendLog($"total elapsed {sw.Elapsed:hh\\:mm\\:ss}");
        AppendLog($"log saved to: {logPath}");
        CloseLogFile();
        _eta.Text = $"done in {sw.Elapsed:hh\\:mm\\:ss}";
        _convert.Enabled = true; _cancel.Enabled = false;
        AnalyzeFolder();   // refresh row statuses from what actually got built
    }

    // %LOCALAPPDATA%/Quarry/logs/convert_<timestamp>.log, with a header. Returns the path.
    private string OpenLogFile()
    {
        string dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Quarry", "logs");
        string path = Path.Combine(dir, $"convert_{DateTime.Now:yyyyMMdd_HHmmss}.log");
        try
        {
            Directory.CreateDirectory(dir);
            lock (_logLock)
            {
                _logFile = new System.IO.StreamWriter(path, append: false) { AutoFlush = true };
                _logFile.WriteLine($"Quarry {QuarryInfo.Version} - convert log");
                _logFile.WriteLine($"started {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
                // Environment goes in the header because these logs are read by someone who does
                // not have the tester's machine: OS build, bitness and free space explain a whole
                // class of failures (a 32-bit process, a full disk) that otherwise look arbitrary.
                _logFile.WriteLine($"system  {Environment.OSVersion.VersionString}, {(Environment.Is64BitProcess ? "64-bit" : "32-bit")} process, .NET {Environment.Version}");
                _logFile.WriteLine($"iso     {_isoBox.Text}");
                _logFile.WriteLine($"out     {_outBox.Text}");
                try
                {
                    var drive = new DriveInfo(Path.GetPathRoot(Path.GetFullPath(_outBox.Text)) ?? "C:\\");
                    _logFile.WriteLine($"space   {drive.AvailableFreeSpace / (1024 * 1024 * 1024)} GB free on {drive.Name}");
                }
                catch { /* unreadable drive: not worth failing a convert over */ }
                _logFile.WriteLine(new string('-', 64));
            }
        }
        catch { /* logging is best-effort; never block a convert */ }
        return path;
    }

    // Open the log folder, selecting the newest convert log so a tester can attach it
    // without hunting. Falls back to opening the folder, then to showing the path as text
    // if the shell refuses - this is a convenience and must never throw into a convert.
    private void OpenLogFolder()
    {
        string dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                                  "Quarry", "logs");
        try
        {
            Directory.CreateDirectory(dir);
            var newest = new DirectoryInfo(dir).GetFiles("convert_*.log")
                                               .OrderByDescending(f => f.LastWriteTimeUtc)
                                               .FirstOrDefault();
            var psi = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "explorer.exe",
                Arguments = newest is null ? $"\"{dir}\"" : $"/select,\"{newest.FullName}\"",
                UseShellExecute = true,
            };
            System.Diagnostics.Process.Start(psi);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, $"Logs are in:{Environment.NewLine}{dir}{Environment.NewLine}{Environment.NewLine}({ex.Message})",
                            "Quarry", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }
    }

    private void CloseLogFile()
    {
        lock (_logLock) { try { _logFile?.Flush(); _logFile?.Dispose(); } catch { } _logFile = null; }
    }

    // Route a log line to the timestamped file (thread-safe) AND the UI log. Called from baker
    // threads via cx.Log, so the file write is locked and the UI write marshals to the UI thread.
    private void AppendLog(string line)
    {
        lock (_logLock) { try { _logFile?.WriteLine($"[{DateTime.Now:HH:mm:ss}] {line}"); } catch { } }
        if (!IsHandleCreated) return;
        if (InvokeRequired)
        {
            try { BeginInvoke(() => _log.AppendText(line + Environment.NewLine)); }
            catch (ObjectDisposedException) { }
            catch (InvalidOperationException) { }
        }
        else _log.AppendText(line + Environment.NewLine);
    }

    // Auto (launch): quietly updates the status label. Manual (button): also pops a
    // message box so the click always gives clear feedback - including "you're up to date".
    private async Task CheckUpdates(bool manual)
    {
        _updStatus.Text = "checking for updates..."; _updStatus.ForeColor = Color.Gray;
        var r = await UpdateChecker.CheckAsync(QuarryInfo.Version, _http, CancellationToken.None);
        if (!IsHandleCreated) return;
        BeginInvoke(() =>
        {
            if (!r.Reachable)
            {
                _updStatus.Text = "couldn't check"; _updStatus.ForeColor = Color.Gray;
                if (manual) MessageBox.Show(this,
                    "Couldn't reach the update server. Check your internet connection and try again.",
                    "Check for updates", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            else if (r.UpToDate)
            {
                _updStatus.Text = "up to date"; _updStatus.ForeColor = Color.DarkGreen;
                if (manual) MessageBox.Show(this,
                    $"You have the latest version ({QuarryInfo.Version}).",
                    "Check for updates", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            else
            {
                _updStatus.Text = $"update available: {r.Latest}"; _updStatus.ForeColor = Color.DarkOrange;
                if (manual)
                {
                    var res = MessageBox.Show(this,
                        $"A newer version is available: {r.Latest}\nYou have {QuarryInfo.Version}.\n\nOpen the download page?",
                        "Update available", MessageBoxButtons.YesNo, MessageBoxIcon.Information);
                    if (res == DialogResult.Yes && r.Url is not null)
                        System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo(r.Url) { UseShellExecute = true });
                }
            }
        });
    }
}
