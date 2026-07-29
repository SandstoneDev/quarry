namespace Quarry;

// Reads the manifest in an existing data/ folder and reports, per section, whether
// it is up to date / needs an update (stage version bumped) / not built / corrupt.
// Non-fatal: any IO/parse failure degrades a stage to NotBuilt.
public static class DataAnalyzer
{
    // One stage's status. currentVersion is the Step.Version in code today.
    public static SectionStatus AnalyzeStage(string stageId, int currentVersion, string dataDir)
    {
        try
        {
            var m = Manifest.Load(dataDir);
            if (!m.Stages.TryGetValue(stageId, out var st) || st.Outputs.Count == 0)
                return SectionStatus.NotBuilt;
            if (st.Version != currentVersion) return SectionStatus.NeedsUpdate;
            foreach (var o in st.Outputs)
            {
                string full = Path.Combine(dataDir, o.Path.Replace('/', Path.DirectorySeparatorChar));
                if (!File.Exists(full)) return SectionStatus.Corrupt;           // recorded but gone
                var fi = new FileInfo(full);
                if (fi.Length != o.Size || Manifest.Sha256(full) != o.Sha256)
                    return SectionStatus.Corrupt;
            }
            return SectionStatus.UpToDate;
        }
        catch { return SectionStatus.NotBuilt; }
    }

    // Combine the manifest-tracked steps of a section into one status. Worst wins,
    // ordered Corrupt > NotBuilt > NeedsUpdate > UpToDate.
    public static SectionAnalysis AnalyzeSection(ConvertPipeline.Section sec, string dataDir, EtaStore eta)
    {
        var tracked = sec.Steps.Where(s => s.StageId.Length > 0).ToArray();
        SectionStatus worst = SectionStatus.UpToDate;
        if (tracked.Length == 0) worst = SectionStatus.NotBuilt;
        foreach (var step in tracked)
        {
            var s = AnalyzeStage(step.StageId, step.Version, dataDir);
            worst = Worse(worst, s);
        }
        return new SectionAnalysis(worst, eta.Estimate(sec.Id));
    }

    private static SectionStatus Worse(SectionStatus a, SectionStatus b)
    {
        int Rank(SectionStatus s) => s switch
        {
            SectionStatus.Corrupt => 3, SectionStatus.NotBuilt => 2,
            SectionStatus.NeedsUpdate => 1, _ => 0
        };
        return Rank(b) > Rank(a) ? b : a;
    }
}
