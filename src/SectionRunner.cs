using System.Threading;
using System.Threading.Tasks;
using static Quarry.ConvertPipeline;
namespace Quarry;

// Runs the shared prepare phase, then each enabled section SEQUENTIALLY, emitting
// SectionEvents (queued -> running -> done/failed/cancelled) for the per-section UI
// rows and recording each section's wall-clock into the EtaStore. Sequential by design
// (v1): the disc reader (one mutable seek position) and the shared TempDir extraction of
// the big archives are not safe to run concurrently; true section-parallelism is a scoped
// follow-up. Per-section failure is isolated: that section -> Failed, the rest still run.
// Cancel is honored between sections AND mid-bake (cx.Ct kills the python process via
// PythonRunner) -> the running section goes Cancelled and the remaining ones are marked
// Cancelled without running. Calls RunSection with snapshotRoots=null (whole-dir diff --
// safe because only one stage runs at a time).
public static class SectionRunner
{
    public static async Task<bool> RunAsync(ConvertContext cx, IReadOnlyList<Section> enabled,
        IProgress<SectionEvent> events, EtaStore eta, CancellationToken ct)
    {
        events.Report(new("__prepare", SectionStatus.Running, 0, "preparing (reading disc)"));
        bool prep = await Task.Run( => RunPrepare(cx), ct);
        if (!prep) { events.Report(new("__prepare", SectionStatus.Failed, 0, "prepare failed")); return false; }

        var manifest = Manifest.Load(cx.OutDir);
        foreach (var s in enabled) events.Report(new(s.Id, SectionStatus.Queued, 0, null));

        bool allOk = true;
        foreach (var sec in enabled)
        {
            if (ct.IsCancellationRequested) { events.Report(new(sec.Id, SectionStatus.Cancelled, 0, null)); continue; }
            // route this section's baker % to its row (sequential -> no race on cx.OnPercent)
            cx.OnPercent = pct => events.Report(new(sec.Id, SectionStatus.Running, pct, null));
            events.Report(new(sec.Id, SectionStatus.Running, 0, $"starting {sec.Id}"));
            var sw = System.Diagnostics.Stopwatch.StartNew;
            bool ok;
            try { ok = await Task.Run( => RunSection(cx, sec, manifest), ct); }
            catch (OperationCanceledException) { ok = false; }
            catch (Exception ex) { events.Report(new(sec.Id, SectionStatus.Failed, 0, ex.Message)); ok = false; }
            finally { sw.Stop; cx.OnPercent = null; }

            if (ct.IsCancellationRequested) events.Report(new(sec.Id, SectionStatus.Cancelled, 0, null));
            else if (ok) { eta.Record(sec.Id, sw.Elapsed); events.Report(new(sec.Id, SectionStatus.Done, 100, $"{sec.Id} done")); }
            else { allOk = false; events.Report(new(sec.Id, SectionStatus.Failed, 0, $"{sec.Id} failed")); }
        }

        // persist the manifest header even on a partial / cancelled run.
        manifest.ConverterVersion = QuarryInfo.Version;
        manifest.DiscElf = cx.Disc?.ElfId ?? "";
        manifest.DiscVer = cx.Disc?.Ver ?? "";
        manifest.Save(cx.OutDir);
        return allOk && !ct.IsCancellationRequested;
    }

    // Test seam: run fake section work under the same sequential + isolation rules.
    public static async Task<Dictionary<string, bool>> RunFakeAsync(
        IEnumerable<(string id, Func<Task> fn)> work,
        IProgress<SectionEvent> events, CancellationToken ct)
    {
        var results = new Dictionary<string, bool>;
        foreach (var w in work)
        {
            if (ct.IsCancellationRequested) break;
            try { await w.fn; results[w.id] = true; events.Report(new(w.id, SectionStatus.Done, 100, null)); }
            catch { results[w.id] = false; events.Report(new(w.id, SectionStatus.Failed, 0, null)); }
        }
        return results;
    }
}
