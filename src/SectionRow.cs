namespace Quarry;

// One export section as a row, laid out by a TableLayoutPanel so nothing overlaps and the
// columns line up across rows: [checkbox + name (fills, ellipsized)] [bar] [badge] [~time].
// Every cell is Dock=Fill with a vertical margin, so the framework centres each control in
// the row height - no manual pixel positioning. Thread-safe setters marshal to the UI thread.
public sealed class SectionRow : UserControl
{
    private const int RowHeight = 30;

    private readonly CheckBox _check = new
    {
        AutoSize = false, Dock = DockStyle.Fill, AutoEllipsis = true,
        TextAlign = ContentAlignment.MiddleLeft, Margin = new Padding(6, 0, 8, 0),
    };
    private readonly ProgressBar _bar = new
    {
        Style = ProgressBarStyle.Continuous, Dock = DockStyle.Fill,
        Margin = new Padding(2, 8, 2, 8),
    };
    private readonly Label _badge = new
    {
        AutoSize = false, Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleCenter,
        Margin = new Padding(2, 5, 2, 5), Font = new Font(FontFamily.GenericSansSerif, 8f),
    };
    private readonly Label _time = new
    {
        AutoSize = false, Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleRight,
        ForeColor = Color.Gray, Margin = new Padding(2, 0, 8, 0),
    };

    public string SectionId { get; }
    public bool Checked { get => _check.Checked; set => _check.Checked = value; }
    public bool CheckEnabled { get => _check.Enabled; set => _check.Enabled = value; }

    public SectionRow(string sectionId, string name, bool defaultOn)
    {
        SectionId = sectionId;
        _check.Text = name; _check.Checked = defaultOn;
        Height = RowHeight; Dock = DockStyle.Top;

        var grid = new TableLayoutPanel
        {
            Dock = DockStyle.Fill, ColumnCount = 4, RowCount = 1, Margin = Padding.Empty,
        };
        grid.RowStyles.Add(new RowStyle(SizeType.Percent, 100f));
        grid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f));   // name -> fills, ellipsizes
        grid.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 104f));  // progress bar
        grid.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 96f));   // status badge
        grid.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 70f));   // ~time (fits "~1h27m")
        grid.Controls.Add(_check, 0, 0);
        grid.Controls.Add(_bar,   1, 0);
        grid.Controls.Add(_badge, 2, 0);
        grid.Controls.Add(_time,  3, 0);
        Controls.Add(grid);

        SetStatus(SectionStatus.NotBuilt);
    }

    public void SetProgress(int pct) => OnUi( => { _bar.Value = Math.Clamp(pct, 0, 100); });
    public void SetEstimate(TimeSpan t) => OnUi( => { _time.Text = t.TotalSeconds < 1 ? "" : Human(t); });

    public void SetStatus(SectionStatus s) => OnUi( =>
    {
        (_badge.Text, _badge.ForeColor, _badge.BackColor) = s switch
        {
            SectionStatus.UpToDate    => ("up to date",   Color.FromArgb(110, 231, 160), Color.FromArgb(30, 58, 42)),
            SectionStatus.NeedsUpdate => ("needs update", Color.FromArgb(231, 201, 110), Color.FromArgb(58, 51, 30)),
            SectionStatus.NotBuilt    => ("not built",    Color.FromArgb(200, 200, 200), Color.FromArgb(58, 58, 58)),
            SectionStatus.Corrupt     => ("corrupt",      Color.FromArgb(231, 138, 138), Color.FromArgb(58, 30, 30)),
            SectionStatus.Queued      => ("queued",       Color.FromArgb(190, 190, 190), Color.FromArgb(48, 48, 48)),
            SectionStatus.Running     => ("baking",       Color.FromArgb(110, 184, 231), Color.FromArgb(30, 46, 58)),
            SectionStatus.Done        => ("done",         Color.FromArgb(110, 231, 160), Color.FromArgb(30, 58, 42)),
            SectionStatus.Failed      => ("failed",       Color.FromArgb(231, 138, 138), Color.FromArgb(58, 30, 30)),
            SectionStatus.Cancelled   => ("cancelled",    Color.FromArgb(190, 190, 190), Color.FromArgb(48, 48, 48)),
            _                         => ("",             Color.Gray,                    Color.Transparent),
        };
        if (s == SectionStatus.Done || s == SectionStatus.UpToDate) _bar.Value = 100;
    });

    private void OnUi(Action a)
    {
        if (!InvokeRequired) { a; return; }          // already on the UI thread
        if (!IsHandleCreated) return;                  // background thread, control not ready -> drop (initial state already set)
        try { BeginInvoke(a); }
        catch (ObjectDisposedException) { /* row/form disposed mid-flight */ }
        catch (InvalidOperationException) { /* handle destroyed between the check and the invoke */ }
    }

    private static string Human(TimeSpan t) =>
        t.TotalMinutes >= 1 ? $"~{Math.Round(t.TotalMinutes)}m" : $"~{Math.Round(t.TotalSeconds)}s";
}
