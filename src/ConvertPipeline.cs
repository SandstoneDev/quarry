// The conversion pipeline: from a disc image to an engine data/ folder. Steps
// are grouped into user-facing Sections (the checkboxes in the UI); a "prepare"
// phase (disc read + plain-file staging) always runs first. phase 2 wires the WORLD
// map bake (the ps2world chain); audio / vehicles / cutscenes / interiors / HUD
// have bakers in tools/ but show as a greyed roadmap until wired.
//
// KEEP THIS LIST IN SYNC WITH THE ENGINE'S BAKERS (tools/*.py). Rule of the
// project: any change to a baker or a data format must be reflected here
// before a release (see docs progress note 2026-07-23).
using System.Threading;

namespace Quarry;

public sealed class ConvertContext
{
    public required string IsoPath { get; init; }
    public required string OutDir { get; init; }        // the final data/ folder
    public required string TempDir { get; init; }       // extraction staging
    public required Action<string> Log { get; init; }
    public Action<int, int>? Progress { get; init; }    // step, totalSteps
    public CancellationToken Ct { get; init; } = CancellationToken.None;
    public Action<int>? OnPercent { get; set; }
    public DiscInfo? Disc { get; set; }
    public Iso9660Reader? Iso { get; set; }
}

public static class ConvertPipeline
{
    // The plain files the engine (or a later baker) consumes directly.
    // ISO path -> staging-relative destination.
    private static readonly (string src, string dst)[] PlainFiles =
    {
        ("SYSTEM.CNF",            "SYSTEM.CNF"),
        ("DATA/TIMECYCP.DAT",     "data/timecycP.dat"),
        ("DATA/HANDLING.CFG",     "data/handling.cfg"),
        ("DATA/WATER.DAT",        "data/water.dat"),
        ("DATA/WATER1.DAT",       "data/water1.dat"),
        ("DATA/INFO.ZON",         "data/info.zon"),
        ("DATA/MAP.ZON",          "data/map.zon"),
        ("DATA/POPCYCLE.DAT",     "data/popcycle.dat"),
        ("DATA/FONTS.DAT",        "data/fonts.dat"),
        ("DATA/MAPS/CULL.IPL",    "data/maps/cull.ipl"),
        ("DATA/SCRIPT/MAIN.SCM",  "data/script/main.scm"),
        ("DATA/SCRIPT/SCRIPT.IMG","data/script/script.img"),
        ("ANIM/PED.IFP",          "anim/ped.ifp"),
    };

    // IMG archives worth a directory listing at phase 1 (extraction comes with the
    // phases that consume them; listing proves the parser against real discs).
    // NOTE: MODELS/GTA3_1.IMG is a byte-duplicate of GTA3.IMG (DVD seek layout) - skip it.
    private static readonly string[] ImgFiles =
    {
        "MODELS/GTA3.IMG",
        "MODELS/PLAYER.IMG",
        "MODELS/GTA_INT.IMG",
        "MODELS/CUTSCENE.IMG",
        "ANIM/CUTS.IMG",
    };

    // A pipeline step. stageId+version drive the incremental manifest: a stage
    // with a stable id whose version and output hashes are unchanged is SKIPPED
    // on re-convert. Setup steps (ISO read, detect) have stageId="" -> always
    // run, never manifest-tracked (they touch no data/ output).
    public readonly record struct Step(string Name, string StageId, int Version,
                                       Func<ConvertContext, bool> Fn);

    // A Section is the user-facing unit (one checkbox in the UI) grouping one or
    // more Steps. Available=false sections render greyed as a roadmap and never
    // run this pass. DefaultOn pre-checks the box.
    public sealed record Section(string Id, string Name, string Desc,
                                 bool DefaultOn, bool Available, Step[] Steps);

    // Prepare phase - always runs first, before any section. Reads the disc and
    // stages the small plain files; touches no data/ output (stageId "").
    private static readonly Step[] PrepareSteps =
    {
        new("Open disc image",    "", 0, StepOpen),
        new("Identify version",   "", 0, StepDetect),
        new("Extract data files", "", 0, StepPlainFiles),
        new("Scan archives",      "", 0, StepScanImgs),
    };

    // The export sections, in run order. Coming-soon entries (Available=false)
    // have bakers in tools/ but are not wired this pass.
    public static readonly Section[] Sections =
    {
        new("core", "Core files",
            "Plain data files the engine reads as-is (timecyc source, script, handling).",
            DefaultOn: true, Available: true, Steps: new[]
            {
                new Step("Stage data folder",   "stage-plain", 1, StepStage),
                new Step("Bake cull-air zones", "cull-air",    1, StepBakeCull),
                new Step("Write boot config",   "bootcfg",     1, StepWriteBootCfg),
            }),
        new("world", "World map (geometry, textures, collision, night, foliage, signs)",
            "The whole world map baked from your disc: geometry, native textures, " +
            "collision, night lighting, grass and road signs. The slow one (~1.5 h).",
            DefaultOn: true, Available: true, Steps: new[]
            {
                new Step("Extract world inputs", "",      0,   StepExtractWorldInputs),
                new Step("Bake world map",       "world", 775, StepBakeWorld),   // v774: the decal classifier now requires SPARSE ink, so baked shadows stop rendering as black patches (v771: ps2_uv_tess caps each triangle's UV extent). Bumping forces a re-bake past the incremental manifest
                new Step("Bake water",           "water", 1,   StepBakeWater),   // sea/lake surface from DATA/water.dat
            }),
        new("timecyc", "Timecycle colours",
            "Sky / ambient / fog colour table for every hour and weather.",
            DefaultOn: true, Available: true, Steps: new[]
            {
                new Step("Bake timecycle", "timecyc", 1, StepBakeTimecyc),
            }),
        new("zones", "Zone names (HUD areas)",
            "The on-screen district names (Ganton, Idlewood, ...).",
            DefaultOn: true, Available: true, Steps: new[]
            {
                new Step("Bake zone names", "zones", 1, StepBakeZones),
            }),
        new("audio", "Audio (SFX, ambience, load tunes)",
            "SFX banks (footsteps, collisions, pain, engines) baked into the engine's VAG " +
            "pool, plus venue ambience and the loading-screen music. Radio/speech are a later pass.",
            DefaultOn: true, Available: true, Steps: new[]
            {
                new Step("Extract audio inputs", "",           0, StepExtractAudioInputs),
                new Step("Bake SFX pool",        "audio-sfx",  2, StepBakeSfx),
                new Step("Bake ambience zones",  "audio-amb",  1, StepBakeAmbience),   // the zone table
                new Step("Bake ambience tracks", "audio-ambx", 3, StepBakeAmbienceTracks), // the audio behind it. v2: the ADPCM frame grid starts 4 bytes after the element header - decoding from the old offset produced noise
                new Step("Bake radio",           "audio-radio",3, StepBakeRadio),      // slow: ~1.5 GB of stations. v2: same 4-byte stream-data offset fix
                new Step("Bake load tunes",      "audio-tune", 1, StepBakeLoadtune),   // non-fatal (only SFX aborts the section)
            }),
        new("vehicles", "Vehicles",
            "Cars, bikes and planes baked from your disc: PS2-native geometry, native textures, " +
            "carcols paint, the damage panels and the embedded collision spheres, plus each model's " +
            "handling. Produces the default car, the whole roster and the model index.",
            DefaultOn: true, Available: true, Steps: new[]
            {
                new Step("Extract vehicle inputs", "",         0, StepExtractVehicleInputs),
                new Step("Bake vehicle roster",    "vehicles", 3, StepBakeVehicles),   // car.bin + veh_index + veh/*.bin (per-vehicle non-fatal). v2: vehicle position scale fixed (was 8x) - bump forces a re-bake past the incremental manifest
                new Step("Bake effects",           "carenv",   2, StepBakeCarEnv),     // non-fatal. v2: +clouds.bin +fxtex.bin (were never baked)
            }),
        // - coming soon: bakers exist in tools/ but are not wired this pass --
        new("cutscenes", "Cutscenes",
            "The intro1a story cutscene: the player-character (CJ) actor, the camera track and " +
            "the subtitles baked from your disc. Big Smoke, the hand props and the voice track " +
            "are PS2-native VIF / VAG and wait on a later codec pass (they degrade gracefully).",
            DefaultOn: true, Available: true, Steps: new[]
            {
                new Step("Extract cutscene inputs", "",          0, StepExtractCutsceneInputs),
                new Step("Bake cutscene models",    "cutscene",  7, StepBakeCutscene),   // cam + csplay/CJ (models degrade). v3: PS2-native skin position scale fixed (actors were 8x) - bump forces a re-bake past the incremental manifest
                new Step("Bake cutscene audio",     "cutaudio",  5, StepBakeCutAudio),   // v2: the voice track now comes off the disc as ADPCM. v3: same 4-byte stream-data offset fix. v4: the take is chosen by subtitle timing, not by length (length picked the wrong scene)
            }),
        new("interiors", "Interiors",
            "Interior world (safehouses, shops, missions) baked from GTA_INT.IMG + the " +
            "interior IPL/IDE maps, plus the entry/exit door markers (enex).",
            DefaultOn: true, Available: true, Steps: new[]
            {
                new Step("Extract interior inputs", "",          0, StepExtractInteriorInputs),
                new Step("Bake enex markers",       "enex",      1, StepBakeEnex),
                new Step("Bake interiors",          "interiors", 2, StepBakeInteriors),   // v2: the PS2 DFF codec kept only the first VIF batch of a multi-batch stream - interiors lost up to 88% of a room's geometry
            }),
        new("hud", "HUD, fonts & menus",
            "HUD sprites, fonts, menu backgrounds, radar and loading screens baked from " +
            "your disc's TXDs (HUD/FONTS/FRONTEN1/2, gta3.img radar tiles).",
            DefaultOn: true, Available: true, Steps: new[]
            {
                new Step("Extract HUD inputs", "",      0, StepExtractHudInputs),
                new Step("Bake HUD & fonts",   "hud",   4, StepBakeHud),   // v4: loading arts come from LOADS<region>.txd (the INTRO files were cutscene backdrops)
                new Step("Bake radar",         "radar", 1, StepBakeRadar),
                new Step("Bake save icon",     "saveicon", 1, StepBakeSaveIcon),   // non-fatal
            }),
        new("peds", "Player & peds (CJ, character, pedestrians)",
            "CJ and the pedestrian models + their animations from your disc. The playable " +
            "hero (skinned CJ + locomotion/idle/fight clips) is baked from PLAYER.IMG + PED.IFP.",
            DefaultOn: true, Available: true, Steps: new[]
            {
                new Step("Extract ped inputs", "",     0, StepExtractPedInputs),  // PLAYER.IMG + PED.IFP (+ gta3.img)
                new Step("Bake hero (CJ)",     "hero", 5, StepBakeHero),          // FATAL: no hero = no playable game
                new Step("Bake player char",   "char", 4, StepBakeChar),          // non-fatal
                new Step("Bake pedestrians",   "peds", 5, StepBakePeds),          // non-fatal (ambient peds are PS2-native). v3: same 8x position-scale fix as the cutscene actors
            }),
        new("effects", "World effects (grass, breakables)",
            "The grass blades the tuft renderer scatters over lawns and fields, and the " +
            "physics table behind every breakable or shovable prop - crates, bins, lamp " +
            "posts, fences.",
            DefaultOn: true, Available: true, Steps: new[]
            {
                new Step("Bake grass blades",    "grass-tex", 1, StepBakeGrassTex),   // non-fatal
                new Step("Bake dynamic objects", "dynobj",    1, StepBakeDynObj),     // non-fatal
                new Step("Bake money pickup",    "money",     1, StepBakeMoney),      // non-fatal
                new Step("Bake map lights",      "lights",    1, StepBakeLights),     // non-fatal, slow (walks every IPL)
            }),
        new("scripts", "Scripts & on-screen text",
            "The script machine: mission triggers, the debug Scripts menu and the on-screen " +
            "hints, plus the string table they print. Both are the demake's own content and " +
            "need nothing from your disc - the disc's own main.scm is a different format the " +
            "engine does not run.",
            DefaultOn: true, Available: true, Steps: new[]
            {
                new Step("Assemble scripts",   "scm",     1, StepBakeScm),
                new Step("Install text table", "strings", 1, StepBakeStrings),
                new Step("Install door marker", "marker", 1, StepBakeMarker),
            }),
    };

    // Run the shared prepare phase (disc open, detect, plain files, scan). Returns false on failure.
    public static bool RunPrepare(ConvertContext cx)
    {
        foreach (var step in PrepareSteps)
        {
            cx.Log($">> {step.Name}");
            if (!step.Fn(cx)) { cx.Log($"!! {step.Name} FAILED"); return false; }
        }
        return true;
    }

    // Run one section's steps sequentially into an already-prepared context. `manifest`
    // is shared; RecordStage+Save run under lock(manifest). `snapshotRoots` scopes the
    // output-diff to this section's owned folders (null = whole data/, for the sequential
    // path). onPercent is forwarded to python bakers. Returns false if a fatal step failed.
    public static bool RunSection(ConvertContext cx, Section sec, Manifest manifest,
                                  string[]? snapshotRoots = null, Action<int>? onPercent = null)
    {
        Dictionary<string, (long size, long ticks)> Snap() =>
            snapshotRoots is null ? Manifest.Snapshot(cx.OutDir)
                                  : Manifest.SnapshotScoped(cx.OutDir, snapshotRoots);
        foreach (var step in sec.Steps)
        {
            cx.Ct.ThrowIfCancellationRequested();
            if (step.StageId.Length > 0 && manifest.CanSkip(step.StageId, step.Version, cx.OutDir, cx.Log))
            {
                cx.Log($"== {sec.Id}/{step.Name}: up to date, skipped");
                continue;
            }
            cx.Log($">> {sec.Id}/{step.Name}");
            var before = step.StageId.Length != 0 ? Snap() : null;
            if (!step.Fn(cx)) { cx.Log($"!! {sec.Id}/{step.Name} FAILED"); return false; }
            if (step.StageId.Length != 0)
            {
                var after = Snap();
                lock (manifest) { manifest.RecordStage(step.StageId, step.Version, cx.OutDir, before!, after); manifest.Save(cx.OutDir); }
            }
        }
        return true;
    }

    /// Run the pipeline. `enabledSections` = the Section.Ids the user ticked; a
    /// section not in the set (or Available=false) is logged and skipped. The
    /// prepare phase always runs first and aborts the whole run on failure.
    public static bool Run(ConvertContext cx, ISet<string> enabledSections)
    {
        var manifest = Manifest.Load(cx.OutDir);
        if (manifest.Stages.Count > 0)
            cx.Log($"existing data/ manifest found (converter {manifest.ConverterVersion}, " +
                   $"disc {manifest.DiscElf} v{manifest.DiscVer}) - unchanged stages will be skipped");

        if (!RunPrepare(cx)) return false;

        // sections: each timed as a whole ("how long each section took"). Sequential
        // path uses whole-dir snapshots (snapshotRoots=null) - safe, one stage at a time.
        var timings = new List<(string name, TimeSpan t)>();
        foreach (var section in Sections)
        {
            if (!section.Available || !enabledSections.Contains(section.Id))
            {
                cx.Log($"== {section.Name}  [skipped]");
                continue;
            }
            cx.Log($"== {section.Name}");
            var sw = System.Diagnostics.Stopwatch.StartNew();
            if (!RunSection(cx, section, manifest)) return false;
            sw.Stop();
            cx.Log($"   done in {sw.Elapsed.TotalSeconds:F1}s");
            timings.Add((section.Name, sw.Elapsed));
        }

        manifest.ConverterVersion = QuarryInfo.Version;
        manifest.DiscElf = cx.Disc?.ElfId ?? "";
        manifest.DiscVer = cx.Disc?.Ver ?? "";
        manifest.Save(cx.OutDir);

        cx.Log("");
        cx.Log("== Timing summary");
        double totalSecs = 0;
        foreach (var (name, t) in timings)
        {
            cx.Log($"   {name,-52} {t.TotalSeconds,7:F1}s");
            totalSecs += t.TotalSeconds;
        }
        cx.Log($"   {"TOTAL",-52} {totalSecs,7:F1}s");
        cx.Log("Done.");
        return true;
    }

    // Run one step's Fn with uniform failure logging. False on return-false or throw.
    private static bool RunStep(ConvertContext cx, Step step)
    {
        try
        {
            if (step.Fn(cx)) return true;
            cx.Log($"FAILED: {step.Name}");
        }
        catch (Exception ex) { cx.Log($"FAILED: {step.Name}: {ex.Message}"); }
        return false;
    }

    private static bool StepOpen(ConvertContext cx)
    {
        cx.Iso = new Iso9660Reader(cx.IsoPath);
        cx.Log($"   ISO9660 ok: {Path.GetFileName(cx.IsoPath)}");
        return true;
    }

    private static bool StepDetect(ConvertContext cx)
    {
        cx.Disc = GameVersion.Probe(cx.Iso!);
        if (cx.Disc is null) { cx.Log("   no SYSTEM.CNF - not a PS2 disc image"); return false; }
        cx.Log($"   {cx.Disc}");
        if (!cx.Disc.Supported)
        {
            cx.Log("   this disc is not in the supported table; conversion may misbehave.");
        }
        return true;
    }

    private static bool StepPlainFiles(ConvertContext cx)
    {
        int got = 0, miss = 0;
        foreach (var (src, dst) in PlainFiles)
        {
            var e = cx.Iso!.Find(src);
            if (e is null) { cx.Log($"   miss {src}"); ++miss; continue; }
            string dest = Path.Combine(cx.TempDir, dst.Replace('/', Path.DirectorySeparatorChar));
            cx.Iso.ExtractTo(e, dest);
            ++got;
        }
        cx.Log($"   {got} extracted, {miss} missing");
        return got > 0;
    }

    private static bool StepScanImgs(ConvertContext cx)
    {
        foreach (var img in ImgFiles)
        {
            var e = cx.Iso!.Find(img);
            if (e is null) { cx.Log($"   miss {img}"); continue; }
            using var s = cx.Iso.OpenRead(e);
            var dir = ImgArchive.ReadDir(s);
            cx.Log($"   {img}: {dir.Count} entries, {e.Size / (1024 * 1024)} MB");
        }
        return true;
    }

    private static bool StepStage(ConvertContext cx)
    {
        // phase 1: the simple layer only - files the engine reads as-is.
        Directory.CreateDirectory(cx.OutDir);
        var stage = new (string tmp, string outRel)[]
        {
            ("data/timecycP.dat",      Path.Combine("world", "timecycP.dat")),
            ("data/script/main.scm",   Path.Combine("script", "main.scm")),
            ("data/handling.cfg",      Path.Combine("vehicles", "handling.cfg")),   // phase 3 vehicle baking input
        };
        int staged = 0;
        foreach (var (tmp, outRel) in stage)
        {
            string src = Path.Combine(cx.TempDir, tmp.Replace('/', Path.DirectorySeparatorChar));
            if (!File.Exists(src)) continue;
            string dst = Path.Combine(cx.OutDir, outRel);
            Directory.CreateDirectory(Path.GetDirectoryName(dst)!);
            File.Copy(src, dst, overwrite: true);
            ++staged;
        }
        cx.Log($"   {staged} file(s) staged into {cx.OutDir}");
        return true;
    }

    // - python baker steps -------------------------------------------------

    private static string? s_python;
    private static bool BakerStep(ConvertContext cx, string script, string srcTmpRel,
                                  string outRel)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null)
        {
            cx.Log("   python not found - step skipped (install Python 3 to enable)");
            return true;                       // phase 1: soft-skip; phase 4 embeds python
        }
        string? sc = PythonRunner.FindScript(script);
        if (sc is null) { cx.Log($"   {script} not found - step skipped"); return true; }
        string src = Path.Combine(cx.TempDir, srcTmpRel.Replace('/', Path.DirectorySeparatorChar));
        if (!File.Exists(src)) { cx.Log($"   {srcTmpRel} not extracted - step skipped"); return true; }
        string dst = Path.Combine(cx.OutDir, outRel);
        Directory.CreateDirectory(Path.GetDirectoryName(dst)!);
        bool ok = PythonRunner.Run(s_python, sc, new[] { src, dst }, cx.Log, null, null, cx.Ct, cx.OnPercent);
        if (!ok) cx.Log($"   {script} FAILED");
        return ok;
    }

    // - scripts section ----------------------------------------------------
    //
    // Unlike every other section these two inputs come from the demake itself, not from
    // the disc, so they ship beside the bakers in content/. The disc's own main.scm is
    // a compiled script in a format this engine does not run; it is staged as a
    // plain file for reference only. The engine reads script/scripts.scm with NO fallback
    // to any other name, so without this step the script machine boots with zero opcodes
    // and every mission trigger, debug Scripts entry and on-screen hint is silently dead.
    private static string? FindContent(string name)
    {
        string exeDir = AppContext.BaseDirectory;
        string[] candidates =
        {
            Path.Combine(exeDir, "content", name),                                    // bundled
            Path.Combine(exeDir, "..", "..", "..", "..", "..", "..", "data", name),   // dev tree
        };
        foreach (var c in candidates)
            if (File.Exists(c)) return Path.GetFullPath(c);
        return null;
    }

    private static bool StepBakeScm(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - step skipped"); return true; }
        string? sc  = PythonRunner.FindScript("scm_asm.py");
        string? src = FindContent("script/scripts.scm.txt");
        if (sc is null)  { cx.Log("   scm_asm.py not found - step skipped"); return true; }
        if (src is null) { cx.Log("   content/script/scripts.scm.txt missing - step skipped"); return true; }

        string dst = Path.Combine(cx.OutDir, "script", "scripts.scm");
        Directory.CreateDirectory(Path.GetDirectoryName(dst)!);
        if (!PythonRunner.Run(s_python, sc, new[] { src, dst }, cx.Log, null, null, cx.Ct, cx.OnPercent))
        {
            cx.Log("   scm_asm.py FAILED");
            return false;
        }
        return true;
    }

    private static bool StepBakeStrings(ConvertContext cx) =>
        CopyContent(cx, "text/strings.txt", Path.Combine("text", "strings.txt"));

    // The entry/exit marker: the translucent yellow cone over every door. enex.bin (the
    // door table) already ships, which is why doors WORK without this - they are just
    // invisible. It is a 534-byte PRP1 blob of our own generated cone geometry with no
    // texture, so like the script listing it travels with the tool.
    private static bool StepBakeMarker(ConvertContext cx) =>
        CopyContent(cx, "effects/marker.bin", Path.Combine("effects", "marker.bin"));

    private static bool CopyContent(ConvertContext cx, string rel, string outRel)
    {
        string? src = FindContent(rel);
        if (src is null) { cx.Log($"   content/{rel} missing - step skipped"); return true; }
        string dst = Path.Combine(cx.OutDir, outRel);
        Directory.CreateDirectory(Path.GetDirectoryName(dst)!);
        File.Copy(src, dst, true);
        cx.Log($"   {outRel.Replace('\\', '/')} <- content/{rel}");
        return true;
    }

    // - world effects -------------------------------------------------------

    // The sea and lake surface. DATA/water.dat is a plain text table staged with the
    // other loose files, so this needs nothing out of the archives.
    private static bool StepBakeWater(ConvertContext cx) =>
        BakerStep(cx, "water_bake.py", "data/water.dat", Path.Combine("world", "water.bin"));

    // Grass blades. On a PS2 disc they live in models/particle.txd (txgrassbig0/1),
    // not in the PC build's plant1.txd, and each is a column of four blade variants.
    private static bool StepBakeGrassTex(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - step skipped"); return true; }
        string? sc = PythonRunner.FindScript("grass_tex_bake.py");
        if (sc is null) { cx.Log("   grass_tex_bake.py not found - step skipped"); return true; }
        string src = Path.Combine(cx.TempDir, "game", "MODELS", "PARTICLE.TXD");
        if (!File.Exists(src)) { cx.Log("   particle.txd not extracted - grass skipped"); return true; }
        string dst = Path.Combine(cx.OutDir, "effects", "grass.bin");
        Directory.CreateDirectory(Path.GetDirectoryName(dst)!);
        if (!PythonRunner.Run(s_python, sc, new[] { src, dst }, cx.Log, null, null, cx.Ct, cx.OnPercent))
            cx.Log("   grass_tex_bake.py FAILED - tufts fall back to untextured");
        return true;                                   // cosmetic: never abort the section
    }

    // Breakable / shovable prop physics from DATA/object.dat, with each model's
    // collision capsule pulled out of gta3.img - so this one needs the archive.
    private static bool StepBakeDynObj(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - step skipped"); return true; }
        string? sc = PythonRunner.FindScript("dynobj_bake.py");
        if (sc is null) { cx.Log("   dynobj_bake.py not found - step skipped"); return true; }
        string saRoot = Path.Combine(cx.TempDir, "game");
        string gta3 = Path.Combine(saRoot, "MODELS", "GTA3.IMG");
        if (!File.Exists(gta3)) { cx.Log("   GTA3.IMG not extracted - dynamic objects skipped"); return true; }
        var env = new Dictionary<string, string> { ["SA_ROOT"] = saRoot, ["SA_GTA3_IMG"] = gta3 };
        string dst = Path.Combine(cx.OutDir, "effects", "dynobj.bin");
        Directory.CreateDirectory(Path.GetDirectoryName(dst)!);
        string names = Path.Combine(cx.TempDir, "dyn_names.txt");
        if (!PythonRunner.Run(s_python, sc, new[] { dst, names }, cx.Log, env, null, cx.Ct, cx.OnPercent))
            cx.Log("   dynobj_bake.py FAILED - breakables stay static");
        return true;                                   // non-fatal: the world still plays
    }

    // One small textured prop out of gta3.img -> a PRP1 blob CProp draws. The model and
    // its dictionary come from the IDE (1212 Money -> dyn_cash, 1277 pickupsave -> icons4).
    private static bool PropStep(ConvertContext cx, string dff, string txd, string outRel, string what)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - step skipped"); return true; }
        string? sc = PythonRunner.FindScript("prop_ps2_bake.py");
        if (sc is null) { cx.Log("   prop_ps2_bake.py not found - step skipped"); return true; }
        string saRoot = Path.Combine(cx.TempDir, "game");
        string gta3 = Path.Combine(saRoot, "MODELS", "GTA3.IMG");
        if (!File.Exists(gta3)) { cx.Log($"   GTA3.IMG not extracted - {what} skipped"); return true; }
        var env = new Dictionary<string, string> { ["SA_ROOT"] = saRoot, ["SA_GTA3_IMG"] = gta3 };
        string dst = Path.Combine(cx.OutDir, outRel);
        Directory.CreateDirectory(Path.GetDirectoryName(dst)!);
        if (!PythonRunner.Run(s_python, sc, new[] { dff, txd, dst }, cx.Log, env, null, cx.Ct, cx.OnPercent))
            cx.Log($"   {what} FAILED - the engine falls back to a flat quad");
        return true;                                   // cosmetic: never abort the section
    }

    // Every 2dfx type-0 light on the map - street lamps, traffic lights, neon - expanded
    // from each IPL placement with the model's own colour, corona size and point-light
    // range. Walks all 43k placements and opens each lit model once, so it is the slowest
    // step outside the world bake.
    private static bool StepBakeLights(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - step skipped"); return true; }
        string? sc = PythonRunner.FindScript("light_bake.py");
        if (sc is null) { cx.Log("   light_bake.py not found - step skipped"); return true; }
        string saRoot = Path.Combine(cx.TempDir, "game");
        string gta3 = Path.Combine(saRoot, "MODELS", "GTA3.IMG");
        if (!File.Exists(gta3)) { cx.Log("   GTA3.IMG not extracted - map lights skipped"); return true; }
        var env = new Dictionary<string, string> { ["SA_ROOT"] = saRoot, ["SA_GTA3_IMG"] = gta3 };
        string dst = Path.Combine(cx.OutDir, "effects", "lights.bin");
        Directory.CreateDirectory(Path.GetDirectoryName(dst)!);
        if (!PythonRunner.Run(s_python, sc, new[] { dst }, cx.Log, env, null, cx.Ct, cx.OnPercent))
            cx.Log("   light_bake.py FAILED - the map stays unlit at night");
        return true;                                   // non-fatal: the world still plays
    }

    private static bool StepBakeMoney(ConvertContext cx) =>
        PropStep(cx, "money", "dyn_cash", Path.Combine("effects", "money.bin"), "money pickup");

    private static bool StepBakeSaveIcon(ConvertContext cx) =>
        PropStep(cx, "pickupsave", "icons4", Path.Combine("hud", "saveicon.bin"), "save icon");

    // - streamed audio (radio, adverts, ambience, the cutscene voice) ----------
    //
    // These live in AUDIO/STREAMS as raw SPU ADPCM behind one global track table. The
    // engine decodes ADPCM directly, so nothing is transcoded: the bake is a copy, it
    // needs no encoder in the bundle, and it lands smaller than the OGG set it replaces.
    private static bool StreamStep(ConvertContext cx, string[] extra, string what, bool slow)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - step skipped"); return true; }
        string? sc = PythonRunner.FindScript("radio_ps2_bake.py");
        if (sc is null) { cx.Log("   radio_ps2_bake.py not found - step skipped"); return true; }
        string audio = Path.Combine(cx.TempDir, "game", "audio");
        if (!Directory.Exists(audio))
            audio = Path.Combine(cx.TempDir, "game", "AUDIO");
        if (!Directory.Exists(audio)) { cx.Log($"   AUDIO/ not extracted - {what} skipped"); return true; }

        // Every mode wants the index, so the station output dir is always passed even
        // when only the ambience or the voice is being pulled; a scratch dir keeps
        // those runs from touching data/audio/radio.
        var args = new List<string> { audio };
        args.AddRange(extra);
        if (slow) cx.Log("   this one copies gigabytes - it is the long step of the section");
        if (!PythonRunner.Run(s_python, sc, args.ToArray(), cx.Log, null, null, cx.Ct, cx.OnPercent))
            cx.Log($"   {what} FAILED - that audio stays silent");
        return true;                                   // never abort: the game plays without it
    }

    // The station packs are ~2 GB and only this step wants them, so they are staged
    // HERE rather than in the shared audio-input step - a convert with the radio
    // switched off never pays for them. Without this the baker died on the first
    // station it could not open (ADVERTS.PAK), which is why a full convert produced
    // "0 stations on the dial": AUDIO/STREAMS/AMBIENCE.PAK was the only pack staged.
    private static bool StageStationPacks(ConvertContext cx)
    {
        string gameRoot = Path.Combine(cx.TempDir, "game");
        int got = 0, reuse = 0;
        long mb = 0;
        foreach (var e in cx.Iso!.ListAll())
        {
            if (e.IsDirectory) continue;
            string p = e.Path;
            if (!p.StartsWith("AUDIO/STREAMS/") || !p.EndsWith(".PAK")) continue;
            string dest = Path.Combine(gameRoot, p.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(dest) && new FileInfo(dest).Length == e.Size) { ++reuse; continue; }
            cx.Iso.ExtractTo(e, dest);
            ++got; mb += e.Size / (1024 * 1024);
            if (cx.Ct.IsCancellationRequested) return false;
        }
        cx.Log($"   {got} station pack(s) staged, {reuse} reused ({mb} MB)");
        return got > 0 || reuse > 0;
    }

    private static bool StepBakeRadio(ConvertContext cx)
    {
        if (!StageStationPacks(cx)) { cx.Log("   no station packs on this disc - radio skipped"); return true; }
        string elf = Path.Combine(cx.TempDir, "game", "SLES_525.41");
        var extra = new List<string> { Path.Combine(cx.OutDir, "audio", "radio") };
        if (File.Exists(elf)) { extra.Add("--elf"); extra.Add(elf); }
        return StreamStep(cx, extra.ToArray(), "radio", true);
    }

    private static bool StepBakeAmbienceTracks(ConvertContext cx) =>
        StreamStep(cx, new[] { Path.Combine(cx.TempDir, "_streams_scratch"),
                               "--ambience", Path.Combine(cx.OutDir, "audio", "amb") },
                   "ambience tracks", false);

    private static bool StepBakeTimecyc(ConvertContext cx) =>
        BakerStep(cx, "timecyc_bin_bake.py", "data/timecycP.dat", "timecyc.bin");

    private static bool StepBakeZones(ConvertContext cx) =>
        BakerStep(cx, "zone_bake.py", "data/info.zon", Path.Combine("hud", "zones.bin"));

    // cull.ipl carries the freeway air-resistance zones baked into data/cull_air.bin.
    // NOTE: despite living under DATA/MAPS/, it is a plain loose ISO file on every
    // real PS2 SA disc checked (EU v1.03 + v2.01) - NOT packed inside MODELS/GTA3.IMG.
    // That archive's own .ipl entries are only the per-zone streaming placements
    // (countryn_stream0.ipl, vegasw_stream3.ipl, ...) plus a handful of mission IPLs
    // (crack.ipl, truthsfarm.ipl, carter.ipl); no "cull" among its 16k+ entries.
    // Staged via PlainFiles like every other always-loaded top-level file (main.scm,
    // handling.cfg, ...), then baked the same way timecyc/zones are.
    private static bool StepBakeCull(ConvertContext cx) =>
        BakerStep(cx, "cull_air_bake.py", "data/maps/cull.ipl", Path.Combine("world", "cull_air.bin"));

    // - world section ------------------------------------------------------

    // Extract the disc's model + map files into an SA_ROOT tree the ps2world
    // chain can read: MODELS/GTA3.IMG + MODELS/GTA_INT.IMG (skip the GTA3_1.IMG
    // byte-dup) and everything under DATA/ except the huge SCRIPT.IMG. gta3.img
    // is ~1 GB -> this is the slow, disk-bound part. stageId "" => always runs
    // when the world section runs (it writes into TempDir, not data/); the bake
    // step's manifest tracks the real .pmap outputs instead.
    private static bool StepExtractWorldInputs(ConvertContext cx)
    {
        string gameRoot = Path.Combine(cx.TempDir, "game");
        var wanted = new List<IsoEntry>();
        foreach (var e in cx.Iso!.ListAll())
        {
            if (e.IsDirectory) continue;
            string p = e.Path;                    // uppercase, '/'-separated
            bool take = p == "MODELS/GTA3.IMG" || p == "MODELS/GTA_INT.IMG" ||
                        (p.StartsWith("DATA/") && p != "DATA/SCRIPT/SCRIPT.IMG");
            if (take) wanted.Add(e);
        }
        long totalMB = wanted.Sum(e => (long)e.Size) / (1024 * 1024);
        cx.Log($"   {wanted.Count} world input file(s), {totalMB} MB (gta3.img is the slow one)");

        int done = 0;
        foreach (var e in wanted)
        {
            string dest = Path.Combine(gameRoot, e.Path.Replace('/', Path.DirectorySeparatorChar));
            if (e.Size > 64L * 1024 * 1024)       // big archive: show byte progress
            {
                cx.Log($"   extracting {e.Path} ({e.Size / (1024 * 1024)} MB)...");
                long nextPct = 25;
                cx.Iso.ExtractTo(e, dest, (d, t) =>
                {
                    long pct = t == 0 ? 100 : d * 100 / t;
                    if (pct >= nextPct) { cx.Log($"      {pct}%"); nextPct += 25; }
                });
            }
            else cx.Iso.ExtractTo(e, dest);
            if (++done % 50 == 0 || done == wanted.Count)
                cx.Log($"   extracted {done}/{wanted.Count} file(s)");
        }
        cx.Log($"   world inputs staged into {gameRoot}");
        return true;
    }

    // The ps2world bake chain - mirrors tools/ps2world_rebake.ps1 exactly (lz4
    // LAST, after the night twin, or it corrupts col/lod/night input). Aborts on
    // any script failure. stageId "world" v716 -> an unchanged re-convert skips
    // this whole ~1.5 h bake.
    private static bool StepBakeWorld(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - world bake skipped (install Python 3)"); return true; }

        string saRoot = Path.Combine(cx.TempDir, "game");
        string gta3   = Path.Combine(saRoot, "MODELS", "GTA3.IMG");
        if (!File.Exists(gta3)) { cx.Log("   world inputs missing - extract step did not run"); return false; }

        var env = new Dictionary<string, string>
        {
            ["SA_ROOT"]     = saRoot,
            ["SA_GTA3_IMG"] = gta3,
        };

        string worldDir = Path.Combine(cx.OutDir, "world", "ps2full");
        string nightDir = worldDir + "_night";
        Directory.CreateDirectory(worldDir);

        // (script, args, fatal) in run order. worldDir/nightDir are the two export trees.
        // FATAL steps produce the load-bearing world (geometry, PSP-sized textures, UV
        // correctness, collision, lz4 the engine reads) -> a failure aborts. Non-fatal steps
        // are ENHANCEMENTS (LOD links, draw-distance, grass, animated/road/render-flag
        // sidecars, night lighting) -> a failure is logged and skipped so a single cosmetic
        // baker hiccup never throws away the ~40 min core bake; the world still loads + plays.
        var chain = new (string script, string[] args, bool fatal)[]
        {
            ("ps2world_pilot.py",     new[] { saRoot, worldDir, "all" },                 true),
            ("pmap_tex_downscale.py", new[] { worldDir, "128", "--road-tier", "128" },   true),
            ("pmap_uv_unsign.py",     new[] { worldDir },                                true),
            ("col_bake.py",           new[] { "regions", worldDir },                     true),
            ("lod_bake_regions.py",   new[] { worldDir },                                false),
            ("pmap_dd_bump.py",       new[] { worldDir, "250" },                         false),
            ("grass_bake.py",         new[] { worldDir },                                false),
            ("dyn_sidecar_bake.py",   new[] { worldDir },                                false),
            ("road_sidecar_bake.py",  new[] { worldDir },                                false),
            ("mflags_sidecar_bake.py",new[] { worldDir },                                false),   // b749/b759: SA IDE render flags (TWOSIDED/ADDITIVE/DRAWLAST/NOZWRITE) -> region_*.mflags; MUST precede lz4 (reads the uncompressed .pmap header)
            ("ps2world_pilot.py",     new[] { saRoot, nightDir, "all", "--night" },      false),
            ("ps2night_sidecar.py",   new[] { nightDir, worldDir },                      false),
            ("pmap_lz4.py",           new[] { "--dir", worldDir },                       true),
        };

        foreach (var (script, args, fatal) in chain)
        {
            cx.Ct.ThrowIfCancellationRequested();
            string? sc = PythonRunner.FindScript(script);
            if (sc is null)
            {
                if (fatal) { cx.Log($"   {script} not found - world bake aborted"); return false; }
                cx.Log($"   {script} not found - skipped (non-fatal enhancement)"); continue;
            }
            cx.Log($"   -> {script} {string.Join(' ', args)}");
            if (!PythonRunner.Run(s_python, sc, args, cx.Log, env, null, cx.Ct, cx.OnPercent))
            {
                cx.Ct.ThrowIfCancellationRequested();   // a cancel-kill is not a "non-fatal skip" -> abort the whole bake
                if (fatal) { cx.Log($"   {script} FAILED - world bake aborted"); return false; }
                cx.Log($"   {script} FAILED - skipped (non-fatal enhancement; world still loads)");
            }
        }

        // chunkset selector for the engine (the pilot already wrote regions.bin).
        File.WriteAllText(Path.Combine(cx.OutDir, "world", "chunkset.txt"), "ps2full");
        // night twin is an intermediate: its colours were folded into the .night
        // sidecars beside the day pmaps - drop it so it isn't shipped or tracked.
        try { if (Directory.Exists(nightDir)) Directory.Delete(nightDir, recursive: true); }
        catch { /* best-effort */ }

        cx.Log($"   world map baked into {worldDir}");
        return true;
    }

    // - HUD section --------------------------------------------------------

    // The disc TXDs the HUD/font/menu bakers read, staged into the same SA_ROOT tree
    // the world chain uses (TempDir/game). DATA/FONTS.DAT carries the font advance
    // widths font_bake reads from SA_ROOT/data/fonts.dat.
    private static readonly string[] HudInputs =
    {
        "MODELS/HUD.TXD",
        "MODELS/FONTS.TXD",
        "MODELS/FRONTEN1.TXD",
        "MODELS/FRONTEN2.TXD",
        // One loading art per file. A missing entry is tolerated (counted, not fatal),
        // so listing more than a given disc carries costs nothing.
        "MODELS/TXD/LOADSUK.TXD",   // the painted loading arts (LOADS<region>.txd)
        "MODELS/TXD/INTRO3.TXD",
        "MODELS/TXD/INTRO4.TXD",
        "DATA/FONTS.DAT",
    };

    // Stage the HUD inputs into TempDir/game. Self-contained: also pulls MODELS/GTA3.IMG
    // (the 144 radar tiles + blip sprites) if the world step hasn't already staged it.
    // stageId "" => always runs with the section (writes into TempDir, not data/).
    private static bool StepExtractHudInputs(ConvertContext cx)
    {
        string gameRoot = Path.Combine(cx.TempDir, "game");
        int got = 0, miss = 0;
        foreach (var p in HudInputs)
        {
            var e = cx.Iso!.Find(p);
            if (e is null) { cx.Log($"   miss {p}"); ++miss; continue; }
            string dest = Path.Combine(gameRoot, p.Replace('/', Path.DirectorySeparatorChar));
            cx.Iso.ExtractTo(e, dest);
            ++got;
        }

        // radar reads gta3.img; the world extract stages the same file into the same
        // tree - reuse it if present, else pull it here so the HUD section stands alone.
        string gta3 = Path.Combine(gameRoot, "MODELS", "GTA3.IMG");
        if (File.Exists(gta3))
            cx.Log("   MODELS/GTA3.IMG already staged (world step) - reused for radar");
        else
        {
            var e = cx.Iso!.Find("MODELS/GTA3.IMG");
            if (e is null) cx.Log("   miss MODELS/GTA3.IMG - radar bake will fail");
            else
            {
                cx.Log($"   extracting MODELS/GTA3.IMG ({e.Size / (1024 * 1024)} MB, radar tiles)...");
                long nextPct = 25;
                cx.Iso.ExtractTo(e, gta3, (d, t) =>
                {
                    long pct = t == 0 ? 100 : d * 100 / t;
                    if (pct >= nextPct) { cx.Log($"      {pct}%"); nextPct += 25; }
                });
                ++got;
            }
        }
        cx.Log($"   {got} HUD input(s) staged into {gameRoot}, {miss} missing");
        return true;
    }

    // Bake the HUD sprites, the four fonts and the menu/loading textures. StepBakeWorld-
    // style: SA_ROOT points the patched bakers at the disc extract; each writes its .bin
    // into cx.OutDir/hud so the manifest snapshot tracks it. Aborts on a baker failure --
    // EXCEPT loadscs: PS2 has no LOADSCS.txd (INTRO*.TXD hold one art each, not the 16
    // named PC arts), so its PC->PS2 loading-art remap is a pending item -> non-fatal.
    private static bool StepBakeHud(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - HUD bake skipped (install Python 3)"); return true; }

        string saRoot = Path.Combine(cx.TempDir, "game");
        if (!File.Exists(Path.Combine(saRoot, "MODELS", "HUD.TXD")))
        { cx.Log("   HUD inputs missing - extract step did not run"); return false; }

        string hudOut = Path.Combine(cx.OutDir, "hud");
        Directory.CreateDirectory(hudOut);

        var env = new Dictionary<string, string> { ["SA_ROOT"] = saRoot };

        // font4 (CFONT_PRICE, the money/clock/zone/banner face) is the game's own thick
        // DISPLAY font, baked from the LOWER block of fonts.txd "font1" by font4_bake.py --
        // extracted straight from the disc, no downloaded TTF, nothing external shipped.

        string loadsTxd = Path.Combine(saRoot, "MODELS", "TXD", "LOADSUK.TXD");

        // (script, args, fatal). args mirror each baker's CLI; hudOut == <data>/hud.
        var chain = new (string script, string[] args, bool fatal)[]
        {
            ("hud_bake.py",      new[] { hudOut },                     true),
            ("font_bake.py",     new[] { "font1", hudOut },            true),
            ("font_bake.py",     new[] { "font2", hudOut },            true),
            ("font3_bake.py",    new[] { hudOut },                     true),
            ("font4_bake.py",    new[] { hudOut },                     true),
            ("menubg_bake.py",   new[] { hudOut },                     true),
            ("frontend_bake.py", new[] { hudOut },                     true),
            ("loadscs_bake.py",  new[] { loadsTxd, Path.Combine(hudOut, "loadscs.bin") }, false),
        };

        foreach (var (script, args, fatal) in chain)
        {
            string? sc = PythonRunner.FindScript(script);
            if (sc is null) { cx.Log($"   {script} not found - HUD bake aborted"); return false; }
            cx.Log($"   -> {script} {string.Join(' ', args)}");
            if (!PythonRunner.Run(s_python, sc, args, cx.Log, env, null, cx.Ct, cx.OnPercent))
            {
                if (fatal) { cx.Log($"   {script} FAILED - HUD bake aborted"); return false; }
                cx.Log($"   {script} FAILED - skipped (loading arts are optional; the engine boots without them)");
            }
        }
        cx.Log($"   HUD & fonts baked into {hudOut}");
        return true;
    }

    // Bake the stitched radar atlas + blip sprites from gta3.img and hud.txd. Split from
    // StepBakeHud (own stageId "radar") because the 144 tile decodes are the slow part
    // and worth skipping on an unchanged re-convert. SA_GTA3_IMG points at the archive.
    private static bool StepBakeRadar(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - radar bake skipped (install Python 3)"); return true; }

        string saRoot = Path.Combine(cx.TempDir, "game");
        string gta3 = Path.Combine(saRoot, "MODELS", "GTA3.IMG");
        if (!File.Exists(gta3)) { cx.Log("   gta3.img missing - extract step did not run"); return false; }

        string hudOut = Path.Combine(cx.OutDir, "hud");
        Directory.CreateDirectory(hudOut);

        var env = new Dictionary<string, string>
        {
            ["SA_ROOT"]     = saRoot,
            ["SA_GTA3_IMG"] = gta3,
        };

        string? sc = PythonRunner.FindScript("radar_bake.py");
        if (sc is null) { cx.Log("   radar_bake.py not found - radar bake aborted"); return false; }
        cx.Log($"   -> radar_bake.py {hudOut}");
        if (!PythonRunner.Run(s_python, sc, new[] { hudOut }, cx.Log, env, null, cx.Ct, cx.OnPercent))
        { cx.Log("   radar_bake.py FAILED - radar bake aborted"); return false; }

        cx.Log($"   radar baked into {hudOut}");
        return true;
    }

    // - interiors section --------------------------------------------------

    // Stage the interior inputs into TempDir/game (the same SA_ROOT tree the world +
    // HUD chains use). GTA_INT.IMG carries the interior DFF/TXD/COL + binary stream
    // IPLs; the DATA/ tree carries gta.dat, the interior IDE/IPL maps and DEFAULT.IDE.
    // gta3.img is also needed - sa_source.open_img() resolves a few interior props /
    // textures out of the main archive. Self-contained: reuse whatever the world/HUD
    // steps already staged, pull the rest. stageId "" => always runs with the section.
    private static bool StepExtractInteriorInputs(ConvertContext cx)
    {
        string gameRoot = Path.Combine(cx.TempDir, "game");

        // GTA_INT.IMG (interior geometry) - the one file only interiors need.
        string gtaInt = Path.Combine(gameRoot, "MODELS", "GTA_INT.IMG");
        if (File.Exists(gtaInt))
            cx.Log("   MODELS/GTA_INT.IMG already staged - reused");
        else
        {
            var e = cx.Iso!.Find("MODELS/GTA_INT.IMG");
            if (e is null) { cx.Log("   miss MODELS/GTA_INT.IMG - interior bake will fail"); return false; }
            cx.Log($"   extracting MODELS/GTA_INT.IMG ({e.Size / (1024 * 1024)} MB)...");
            cx.Iso.ExtractTo(e, gtaInt);
        }

        // GTA3.IMG (main archive) - reused if the world/HUD step already pulled it;
        // else pulled here so interiors can run standalone (it is the ~1 GB slow one).
        string gta3 = Path.Combine(gameRoot, "MODELS", "GTA3.IMG");
        if (File.Exists(gta3))
            cx.Log("   MODELS/GTA3.IMG already staged - reused");
        else
        {
            var e = cx.Iso!.Find("MODELS/GTA3.IMG");
            if (e is null) cx.Log("   miss MODELS/GTA3.IMG - interior props in the main archive will be absent");
            else
            {
                cx.Log($"   extracting MODELS/GTA3.IMG ({e.Size / (1024 * 1024)} MB)...");
                long nextPct = 25;
                cx.Iso.ExtractTo(e, gta3, (d, t) =>
                {
                    long pct = t == 0 ? 100 : d * 100 / t;
                    if (pct >= nextPct) { cx.Log($"      {pct}%"); nextPct += 25; }
                });
            }
        }

        // DATA/ tree (gta.dat + interior IDE/IPL + DEFAULT.IDE), minus the huge
        // SCRIPT.IMG. Skip files already staged by the world step.
        int got = 0, skip = 0;
        foreach (var e in cx.Iso!.ListAll())
        {
            if (e.IsDirectory) continue;
            string p = e.Path;                    // uppercase, '/'-separated
            if (!p.StartsWith("DATA/") || p == "DATA/SCRIPT/SCRIPT.IMG") continue;
            string dest = Path.Combine(gameRoot, p.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(dest)) { ++skip; continue; }
            cx.Iso.ExtractTo(e, dest);
            ++got;
        }
        cx.Log($"   DATA/ tree: {got} extracted, {skip} already staged");
        cx.Log($"   interior inputs staged into {gameRoot}");
        return true;
    }

    // Bake the entry/exit door markers -> interiors/enex.bin. Pure TEXT parse of the
    // disc's IPL enex sections (no codec); SA_ROOT points enex_bake at the extract.
    private static bool StepBakeEnex(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - enex bake skipped (install Python 3)"); return true; }

        string saRoot = Path.Combine(cx.TempDir, "game");
        if (!Directory.Exists(Path.Combine(saRoot, "DATA", "MAPS")))
        { cx.Log("   DATA/MAPS missing - extract step did not run"); return false; }

        string intOut = Path.Combine(cx.OutDir, "interiors");
        Directory.CreateDirectory(intOut);

        var env = new Dictionary<string, string> { ["SA_ROOT"] = saRoot };

        string? sc = PythonRunner.FindScript("enex_bake.py");
        if (sc is null) { cx.Log("   enex_bake.py not found - enex bake aborted"); return false; }
        cx.Log($"   -> enex_bake.py {intOut}");
        if (!PythonRunner.Run(s_python, sc, new[] { intOut }, cx.Log, env, null, cx.Ct, cx.OnPercent))
        { cx.Log("   enex_bake.py FAILED - enex bake aborted"); return false; }

        cx.Log($"   enex markers baked into {intOut}");
        return true;
    }

    // Bake every enterable interior -> interiors/interior_<name>.pmap + .col via the
    // all-interiors driver (like the world chain drives ps2world_pilot). StepBakeWorld-
    // style: SA_ROOT points the PS2-codec-swapped bakers at the disc extract; the driver
    // writes into cx.OutDir/interiors so the manifest snapshot tracks the outputs.
    private static bool StepBakeInteriors(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - interior bake skipped (install Python 3)"); return true; }

        string saRoot = Path.Combine(cx.TempDir, "game");
        string gtaInt = Path.Combine(saRoot, "MODELS", "GTA_INT.IMG");
        if (!File.Exists(gtaInt)) { cx.Log("   GTA_INT.IMG missing - extract step did not run"); return false; }

        string intOut = Path.Combine(cx.OutDir, "interiors");
        Directory.CreateDirectory(intOut);

        var env = new Dictionary<string, string> { ["SA_ROOT"] = saRoot };

        string? sc = PythonRunner.FindScript("bake_all_interiors.py");
        if (sc is null) { cx.Log("   bake_all_interiors.py not found - interior bake aborted"); return false; }
        cx.Log($"   -> bake_all_interiors.py {intOut}");
        if (!PythonRunner.Run(s_python, sc, new[] { intOut }, cx.Log, env, null, cx.Ct, cx.OnPercent))
        { cx.Log("   bake_all_interiors.py FAILED - interior bake aborted"); return false; }

        cx.Log($"   interiors baked into {intOut}");
        return true;
    }

    // - audio section ------------------------------------------------------

    // The SFX bank paks actually baked into the pool (footsteps / collisions / pain /
    // engines / loading). SCRIPT* + SPC_* (mission-script + character speech, hundreds of
    // MB) are NOT baked -> skipped at extract. The disc ships each pak as '<base>01.pak'
    // plus an '02.pak' byte-duplicate for the DVD seek layout; sa_audio resolves either.
    private static readonly string[] SfxPakPrefixes = { "FEET", "GENRL", "PAIN_A" };

    private static bool WantSfxPak(string isoPath)   // isoPath e.g. "AUDIO/SFX/GENRL01.PAK"
    {
        string name = isoPath.Substring("AUDIO/SFX/".Length);
        foreach (var pre in SfxPakPrefixes)
            if (name.StartsWith(pre, StringComparison.OrdinalIgnoreCase)) return true;
        return false;
    }

    // Stage the audio inputs into TempDir/game (the same SA_ROOT tree world/HUD/interiors
    // use): AUDIO/CONFIG/*.DAT (bank + pak lookups, a few KB) + the baked SFX paks + the
    // ambience map DATA/MAPS/AUDIOZON.IPL + AUDIO/STREAMS/AMBIENCE.PAK (the one ambience
    // stream; the other ~2 GB of STREAMS is radio / speech / cutscene -> deferred to phase 4).
    // stageId "" => always runs with the section (writes into TempDir, not data/).
    private static bool StepExtractAudioInputs(ConvertContext cx)
    {
        string gameRoot = Path.Combine(cx.TempDir, "game");
        int got = 0, reuse = 0;
        foreach (var e in cx.Iso!.ListAll())
        {
            if (e.IsDirectory) continue;
            string p = e.Path;                        // uppercase, '/'-separated
            bool take = p.StartsWith("AUDIO/CONFIG/") ||
                        (p.StartsWith("AUDIO/SFX/") && WantSfxPak(p)) ||
                        p == "DATA/MAPS/AUDIOZON.IPL" ||
                        p == "AUDIO/STREAMS/AMBIENCE.PAK";
            if (!take) continue;
            string dest = Path.Combine(gameRoot, p.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(dest)) { ++reuse; continue; }   // e.g. AUDIOZON.IPL already staged by interiors
            if (e.Size > 32L * 1024 * 1024)
                cx.Log($"   extracting {p} ({e.Size / (1024 * 1024)} MB)...");
            cx.Iso.ExtractTo(e, dest);
            ++got;
        }
        cx.Log($"   {got} audio input file(s) staged into {gameRoot} ({reuse} reused)");
        return got > 0 || reuse > 0;
    }

    // Bake the SFX pool -> data/audio/sfx.bin (PRIMARY). PS2 disc bodies are already native
    // Sony PS-ADPCM (VAG), so audio_bake passes them straight into the engine's VAG pool --
    // no transcode. QUARRY_SFX_NO_JINGLE keeps the bake on the Python stdlib (the BEATS
    // mission jingle is radio territory, phase 4). This is the one audio step that ABORTS the
    // section on failure. SA_ROOT points the patched baker at the disc extract.
    private static bool StepBakeSfx(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - SFX bake skipped (install Python 3)"); return true; }

        string saRoot = Path.Combine(cx.TempDir, "game");
        if (!Directory.Exists(Path.Combine(saRoot, "AUDIO", "CONFIG")))
        { cx.Log("   AUDIO/CONFIG missing - extract step did not run"); return false; }

        var env = new Dictionary<string, string>
        {
            ["SA_ROOT"]              = saRoot,
            ["QUARRY_SFX_NO_JINGLE"] = "1",   // BEATS mission jingle -> radio pass (phase 4); keeps sfx stdlib-only
        };

        string? sc = PythonRunner.FindScript("audio_bake.py");
        if (sc is null) { cx.Log("   audio_bake.py not found - SFX bake aborted"); return false; }
        cx.Log($"   -> audio_bake.py {cx.OutDir}");
        if (!PythonRunner.Run(s_python, sc, new[] { cx.OutDir }, cx.Log, env, null, cx.Ct, cx.OnPercent))
        { cx.Log("   audio_bake.py FAILED - SFX bake aborted"); return false; }

        string sfx = Path.Combine(cx.OutDir, "audio", "sfx.bin");
        if (!File.Exists(sfx) || new FileInfo(sfx).Length < 64)
        { cx.Log("   sfx.bin missing/empty - SFX bake aborted"); return false; }
        cx.Log($"   SFX pool baked -> {sfx} ({new FileInfo(sfx).Length / 1024} KB)");
        return true;
    }

    // Bake venue ambience -> data/audio/amb/ (ambzones.bin + amb_t*.ogg). NON-FATAL: the OGG
    // transcode needs ffmpeg (imageio_ffmpeg) and PC-style OGG streams; on the PS2 disc the
    // streams are VAG, so ambience_bake soft-skips the audio (still writing the stdlib-only
    // ambzones.bin map) and this step never aborts the section.
    private static bool StepBakeAmbience(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - ambience skipped"); return true; }

        string saRoot = Path.Combine(cx.TempDir, "game");
        var env = new Dictionary<string, string> { ["SA_ROOT"] = saRoot };

        string? sc = PythonRunner.FindScript("ambience_bake.py");
        if (sc is null) { cx.Log("   ambience_bake.py not found - skipped"); return true; }
        cx.Log($"   -> ambience_bake.py {cx.OutDir}");
        if (!PythonRunner.Run(s_python, sc, new[] { cx.OutDir }, cx.Log, env, null, cx.Ct, cx.OnPercent))
            cx.Log("   ambience_bake.py returned non-zero - skipped (ffmpeg/streams optional, phase 4)");
        return true;   // ambience never aborts the section
    }

    // Bake the 4 loading-screen tunes -> data/audio/loadtune0..3.wav. PS2 bank 82 is native
    // VAG -> loadtune_bake decodes it to PCM (stdlib) before writing the stereo WAVs. Only the
    // SFX step aborts the section, so a loadtune failure is logged but non-fatal.
    private static bool StepBakeLoadtune(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - load tunes skipped"); return true; }

        string saRoot = Path.Combine(cx.TempDir, "game");
        if (!Directory.Exists(Path.Combine(saRoot, "AUDIO", "CONFIG")))
        { cx.Log("   AUDIO/CONFIG missing - load tunes skipped"); return true; }

        var env = new Dictionary<string, string> { ["SA_ROOT"] = saRoot };

        string? sc = PythonRunner.FindScript("loadtune_bake.py");
        if (sc is null) { cx.Log("   loadtune_bake.py not found - skipped"); return true; }
        cx.Log($"   -> loadtune_bake.py {cx.OutDir}");
        if (!PythonRunner.Run(s_python, sc, new[] { cx.OutDir }, cx.Log, env, null, cx.Ct, cx.OnPercent))
        { cx.Log("   loadtune_bake.py FAILED - skipped (non-fatal)"); return true; }
        cx.Log($"   load tunes baked into {Path.Combine(cx.OutDir, "audio")}");
        return true;
    }

    // - vehicles section ---------------------------------------------------

    // Stage the vehicle inputs into TempDir/game (the SA_ROOT tree the other sections share):
    //   MODELS/GTA3.IMG            - the vehicle DFF + per-model TXD source (~1 GB; reused if any
    //                                 earlier section already pulled it)
    //   MODELS/GENERIC/VEHICLE.TXD - the SHARED vehicle textures (generic/grunge/lights/tyres/
    //                                 plates/env) every model falls back to; staged by NO other
    //                                 section, so it is pulled here
    //   MODELS/PARTICLE.TXD        - headlight sprite, cloud + corona art (effects step)
    //   MODELS/EFFECTS.TXD         - the smoke and fireball particles (effects step)
    //   DATA/CARCOLS.DAT + VEHICLES.IDE + HANDLING.CFG - paint combos, the roster and the handling
    //                                 columns (tiny; reused if the world/interior step staged DATA)
    // stageId "" => always runs with the section (writes into TempDir, not data/).
    private static bool StepExtractVehicleInputs(ConvertContext cx)
    {
        string gameRoot = Path.Combine(cx.TempDir, "game");
        int got = 0, reuse = 0, miss = 0;

        // GTA3.IMG - REQUIRED. Reused if world/HUD/interiors/peds pulled it (the ~1 GB slow one).
        string gta3 = Path.Combine(gameRoot, "MODELS", "GTA3.IMG");
        if (File.Exists(gta3)) { cx.Log("   MODELS/GTA3.IMG already staged - reused"); ++reuse; }
        else
        {
            var e = cx.Iso!.Find("MODELS/GTA3.IMG");
            if (e is null) { cx.Log("   miss MODELS/GTA3.IMG - vehicle bake will fail"); return false; }
            cx.Log($"   extracting MODELS/GTA3.IMG ({e.Size / (1024 * 1024)} MB)...");
            long nextPct = 25;
            cx.Iso.ExtractTo(e, gta3, (d, t) =>
            {
                long pct = t == 0 ? 100 : d * 100 / t;
                if (pct >= nextPct) { cx.Log($"      {pct}%"); nextPct += 25; }
            });
            ++got;
        }

        // The vehicle-only inputs: the shared VEHICLE.TXD (not staged elsewhere), the PARTICLE.TXD
        // headlight sprite, and the three DATA text files. Reuse whatever is already staged.
        var wanted = new[]
        {
            "MODELS/GENERIC/VEHICLE.TXD",
            "MODELS/PARTICLE.TXD",
            "MODELS/EFFECTS.TXD",       // smoke + fireball for fxtex, clouds come from PARTICLE.TXD
            "DATA/CARCOLS.DAT",
            "DATA/VEHICLES.IDE",
            "DATA/HANDLING.CFG",
        };
        foreach (var p in wanted)
        {
            string dest = Path.Combine(gameRoot, p.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(dest)) { ++reuse; continue; }
            var e = cx.Iso!.Find(p);
            if (e is null) { cx.Log($"   miss {p}"); ++miss; continue; }
            cx.Iso.ExtractTo(e, dest);
            ++got;
        }
        cx.Log($"   vehicle inputs staged into {gameRoot} ({got} extracted, {reuse} reused, {miss} missing)");
        return File.Exists(gta3);
    }

    // Bake the vehicle roster into <data>/vehicles. StepBakeWorld-style: SA_ROOT/SA_GTA3_IMG point
    // the PS2-codec-swapped car_bake at the disc extract; `--out` redirects every writer into
    // <OutDir>/vehicles (car.bin default single, veh_index.bin, veh/veh_<name>.bin) and drops the
    // dev-loop memstick mirror. car_bake decodes the PS2-native DFF (ps2dff - the world/interior
    // swap), the PS2 TXDs, the carcols paint, the _ok/_dam damage panels and the embedded COL
    // spheres. A per-vehicle decode failure is non-fatal INSIDE the roster driver (it indexes only
    // the models that baked); this step fails only if car_bake itself errors or the two headline
    // outputs (car.bin + veh_index.bin) are missing.
    private static bool StepBakeVehicles(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - vehicle bake skipped (install Python 3 + Pillow)"); return true; }

        string saRoot = Path.Combine(cx.TempDir, "game");
        string gta3 = Path.Combine(saRoot, "MODELS", "GTA3.IMG");
        if (!File.Exists(gta3)) { cx.Log("   gta3.img missing - extract step did not run"); return false; }

        string vehOut = Path.Combine(cx.OutDir, "vehicles");
        Directory.CreateDirectory(vehOut);
        var env = new Dictionary<string, string>
        {
            ["SA_ROOT"]     = saRoot,
            ["SA_GTA3_IMG"] = gta3,
        };

        string? sc = PythonRunner.FindScript("car_bake.py");
        if (sc is null) { cx.Log("   car_bake.py not found - vehicle bake aborted"); return false; }

        // 1) the default single car.bin (bravura, back-compat - Vehicle.c car.bin load path).
        cx.Log($"   -> car_bake.py --out {vehOut}   (default car.bin)");
        if (!PythonRunner.Run(s_python, sc, new[] { "--out", vehOut }, cx.Log, env, null, cx.Ct, cx.OnPercent))
        { cx.Log("   car_bake.py (car.bin) FAILED - vehicle bake aborted"); return false; }

        // 2) the full roster + veh_index.bin (the slow part, ~200 models). Per-vehicle failures are
        //    non-fatal: the driver catches each and indexes only what actually baked.
        cx.Log($"   -> car_bake.py --all --out {vehOut}   (roster + index; the slow one)");
        if (!PythonRunner.Run(s_python, sc, new[] { "--all", "--out", vehOut }, cx.Log, env, null, cx.Ct, cx.OnPercent))
        { cx.Log("   car_bake.py --all FAILED - vehicle bake aborted"); return false; }

        string carBin = Path.Combine(vehOut, "car.bin");
        string idxBin = Path.Combine(vehOut, "veh_index.bin");
        if (!File.Exists(carBin) || new FileInfo(carBin).Length < 64)
        { cx.Log("   car.bin missing/empty - vehicle bake aborted"); return false; }
        if (!File.Exists(idxBin) || new FileInfo(idxBin).Length < 8)
        { cx.Log("   veh_index.bin missing/empty - vehicle bake aborted"); return false; }
        string vehDir = Path.Combine(vehOut, "veh");
        int nveh = Directory.Exists(vehDir) ? Directory.GetFiles(vehDir, "veh_*.bin").Length : 0;
        cx.Log($"   vehicles baked into {vehOut} ({nveh} roster models + car.bin + veh_index.bin)");
        return true;
    }

    // boot.txt sits NEXT TO the EBOOT, not inside data/: it is read before the data
    // folder is known. It carries the boot-time switches. Written only when absent, so a
    // re-convert never overwrites settings someone changed.
    //
    // master_host is the project's own master/relay server. It ships filled in on purpose:
    // multiplayer does not work without it, and the engine only brings the net stack up
    // when this field is set. Point it elsewhere to run your own.
    private const string DefaultBootCfg = """
        me=0
        noaudio=0
        skip_intro=0
        vramext=0
        scm=1
        scm_trace=0
        ttylog=0
        psplink=0
        master_host=134.209.88.211
        master_port=7778
        """;

    private static bool StepWriteBootCfg(ConvertContext cx)
    {
        var parent = Directory.GetParent(cx.OutDir.TrimEnd(Path.DirectorySeparatorChar,
                                                           Path.AltDirectorySeparatorChar));
        if (parent is null) { cx.Log("   cannot resolve the folder above data/ - boot.txt skipped"); return true; }
        string path = Path.Combine(parent.FullName, "boot.txt");
        if (File.Exists(path)) { cx.Log($"   boot.txt already present - left as it is ({path})"); return true; }
        var lines = DefaultBootCfg.Split('\n').Select(l => l.Trim()).Where(l => l.Length > 0);
        File.WriteAllText(path, string.Join("\n", lines) + "\n");
        cx.Log($"   wrote boot config -> {path}");
        return true;
    }

    // Bake the vehicle env-map + headlight projection sprites -> <data>/effects/{carenv,headlight}.bin
    // from generic/vehicle.txd + particle.txd. NON-FATAL: these drive the skygfx vehicle env/spec and
    // twin-spot headlight additive passes; a miss degrades the look, not the drive.
    private static bool StepBakeCarEnv(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - vehicle effects skipped"); return true; }

        string saRoot = Path.Combine(cx.TempDir, "game");
        var env = new Dictionary<string, string> { ["SA_ROOT"] = saRoot };

        string fxDir = Path.Combine(cx.OutDir, "effects");
        Directory.CreateDirectory(fxDir);
        string particle = Path.Combine(saRoot, "MODELS", "PARTICLE.TXD");
        string effects  = Path.Combine(saRoot, "MODELS", "EFFECTS.TXD");

        // All three write into <data>/effects and are non-fatal: without them the game
        // still plays, it just loses the look. Until now only the first was wired, so
        // clouds.bin and fxtex.bin were never produced - the sky had no clouds and
        // CarFx logged "no fxtex.bin -> flat quads", which is the white squares where
        // smoke and fire belong.
        var chain = new (string script, string[] args)[]
        {
            ("carenv_bake.py", new[] { cx.OutDir }),
            ("cloud_bake.py",  new[] { particle, Path.Combine(fxDir, "clouds.bin") }),
            ("fxtex_bake.py",  new[] { effects,  Path.Combine(fxDir, "fxtex.bin")  }),
        };
        foreach (var (script, args) in chain)
        {
            cx.Ct.ThrowIfCancellationRequested();
            string? sc = PythonRunner.FindScript(script);
            if (sc is null) { cx.Log($"   {script} not found - skipped"); continue; }
            cx.Log($"   -> {script} {string.Join(' ', args)}");
            if (!PythonRunner.Run(s_python, sc, args, cx.Log, env, null, cx.Ct, cx.OnPercent))
                cx.Log($"   {script} FAILED - skipped (non-fatal; costs looks, not play)");
        }
        cx.Log($"   effects baked into {fxDir}");
        return true;
    }

    // - peds section -------------------------------------------------------

    // Stage the ped inputs into TempDir/game (the SA_ROOT tree world/HUD/interiors/audio
    // share): PLAYER.IMG (the CJ hero + char components - platform-neutral skinned DFFs,
    // byte-identical to PC; only the TXDs are PS2-native, handled by the baker), ANIM/PED.IFP
    // (base locomotion/idle/fight clips, byte-exact ANP3), and GTA3.IMG (the ambient ped
    // models). PED.IFP goes into the game tree so hero_bake's SA_ROOT-relative sa_ifp finds
    // it. Reuse whatever earlier sections already staged. stageId "" => always runs.
    private static bool StepExtractPedInputs(ConvertContext cx)
    {
        string gameRoot = Path.Combine(cx.TempDir, "game");
        int got = 0, reuse = 0;

        // PLAYER.IMG - REQUIRED (the CJ hero). Small (~30 MB).
        string player = Path.Combine(gameRoot, "MODELS", "PLAYER.IMG");
        if (File.Exists(player)) { cx.Log("   MODELS/PLAYER.IMG already staged - reused"); ++reuse; }
        else
        {
            var e = cx.Iso!.Find("MODELS/PLAYER.IMG");
            if (e is null) { cx.Log("   miss MODELS/PLAYER.IMG - hero bake will fail"); return false; }
            cx.Log($"   extracting MODELS/PLAYER.IMG ({e.Size / (1024 * 1024)} MB)...");
            cx.Iso.ExtractTo(e, player); ++got;
        }

        // ANIM/PED.IFP - REQUIRED for the hero clips. Into the game tree (sa_ifp reads
        // SA_ROOT/anim/ped.ifp). Also staged loose by the core section, but that lives
        // under TempDir, not TempDir/game - so stage it here too.
        string pedifp = Path.Combine(gameRoot, "ANIM", "PED.IFP");
        if (File.Exists(pedifp)) { cx.Log("   ANIM/PED.IFP already staged - reused"); ++reuse; }
        else
        {
            var e = cx.Iso!.Find("ANIM/PED.IFP");
            if (e is null) cx.Log("   miss ANIM/PED.IFP - hero animation clips will be absent");
            else { cx.Iso.ExtractTo(e, pedifp); ++got; }
        }

        // GTA3.IMG - the ambient ped models. Reused if world/HUD/interiors pulled it;
        // else pulled here so peds can run standalone (the ~1 GB slow one).
        string gta3 = Path.Combine(gameRoot, "MODELS", "GTA3.IMG");
        if (File.Exists(gta3)) { cx.Log("   MODELS/GTA3.IMG already staged - reused"); ++reuse; }
        else
        {
            var e = cx.Iso!.Find("MODELS/GTA3.IMG");
            if (e is null) cx.Log("   miss MODELS/GTA3.IMG - ambient peds absent");
            else
            {
                cx.Log($"   extracting MODELS/GTA3.IMG ({e.Size / (1024 * 1024)} MB)...");
                long nextPct = 25;
                cx.Iso.ExtractTo(e, gta3, (d, t) =>
                {
                    long pct = t == 0 ? 100 : d * 100 / t;
                    if (pct >= nextPct) { cx.Log($"      {pct}%"); nextPct += 25; }
                });
                ++got;
            }
        }
        cx.Log($"   ped inputs staged into {gameRoot} ({got} extracted, {reuse} reused)");
        return File.Exists(player);
    }

    // Bake the playable hero -> peds/hero.bin (CRITICAL: no hero = no playable game).
    // StepBakeWorld-style: SA_ROOT points hero_bake at the disc extract; it reads the
    // PLAYER.IMG skinned CJ components (37-bone skin + HAnim), the PS2 TXDs, and the
    // PED.IFP clips, and writes the HRO2 stream straight into <data>/peds/hero.bin (the
    // baker skips its dev-loop memstick mirror when handed an explicit output path).
    private static bool StepBakeHero(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - hero bake skipped (install Python 3 + Pillow)"); return true; }

        string saRoot = Path.Combine(cx.TempDir, "game");
        if (!File.Exists(Path.Combine(saRoot, "MODELS", "PLAYER.IMG")))
        { cx.Log("   PLAYER.IMG missing - extract step did not run"); return false; }

        string pedsOut = Path.Combine(cx.OutDir, "peds");
        Directory.CreateDirectory(pedsOut);
        var env = new Dictionary<string, string> { ["SA_ROOT"] = saRoot };

        string? sc = PythonRunner.FindScript("hero_bake.py");
        if (sc is null) { cx.Log("   hero_bake.py not found - hero bake aborted"); return false; }
        string dst = Path.Combine(pedsOut, "hero.bin");
        cx.Log($"   -> hero_bake.py cj {dst}");
        if (!PythonRunner.Run(s_python, sc, new[] { "cj", dst }, cx.Log, env, null, cx.Ct, cx.OnPercent))
        { cx.Log("   hero_bake.py FAILED - hero bake aborted"); return false; }

        if (!File.Exists(dst) || new FileInfo(dst).Length < 1024)
        { cx.Log("   hero.bin missing/empty - hero bake aborted"); return false; }
        cx.Log($"   hero (CJ) baked -> {dst} ({new FileInfo(dst).Length / 1024} KB)");
        return true;
    }

    // Bake the resident player-character model -> peds/char.bin. NON-FATAL: char.bin is a
    // secondary bind-pose model; the hero (above) is what the game actually plays.
    private static bool StepBakeChar(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - char bake skipped"); return true; }

        string saRoot = Path.Combine(cx.TempDir, "game");
        if (!File.Exists(Path.Combine(saRoot, "MODELS", "PLAYER.IMG")))
        { cx.Log("   PLAYER.IMG missing - char bake skipped"); return true; }

        string pedsOut = Path.Combine(cx.OutDir, "peds");
        Directory.CreateDirectory(pedsOut);
        var env = new Dictionary<string, string> { ["SA_ROOT"] = saRoot };

        string? sc = PythonRunner.FindScript("char_bake.py");
        if (sc is null) { cx.Log("   char_bake.py not found - char bake skipped"); return true; }
        string dst = Path.Combine(pedsOut, "char.bin");
        cx.Log($"   -> char_bake.py cj {dst}");
        if (!PythonRunner.Run(s_python, sc, new[] { "cj", dst }, cx.Log, env, null, cx.Ct, cx.OnPercent))
        { cx.Log("   char_bake.py FAILED - skipped (non-fatal)"); return true; }
        cx.Log($"   player char baked -> {dst}");
        return true;
    }

    // Bake the ambient pedestrians -> peds/peds.bin. NON-FATAL. On a PS2 disc the ambient
    // ped models (gta3.img) are PS2-NATIVE skinned DFFs (native VIF geometry + native skin
    // plugin) - a codec the port does not have yet, so ped_bake skips each and writes a
    // valid empty 'PEDS' container (the engine treats it as hero-only). The hero + char
    // above carry the playable game; ambient peds are a later pass (see the ped-baker note).
    private static bool StepBakePeds(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - peds bake skipped"); return true; }

        string saRoot = Path.Combine(cx.TempDir, "game");
        var env = new Dictionary<string, string> { ["SA_ROOT"] = saRoot };

        string pedsOut = Path.Combine(cx.OutDir, "peds");
        Directory.CreateDirectory(pedsOut);

        string? sc = PythonRunner.FindScript("ped_bake.py");
        if (sc is null) { cx.Log("   ped_bake.py not found - peds bake skipped"); return true; }
        string dst = Path.Combine(pedsOut, "peds.bin");
        cx.Log($"   -> ped_bake.py {dst}");
        if (!PythonRunner.Run(s_python, sc, new[] { dst }, cx.Log, env, null, cx.Ct, cx.OnPercent))
        { cx.Log("   ped_bake.py FAILED - skipped (non-fatal)"); return true; }
        cx.Log($"   pedestrians baked -> {dst}");
        return true;
    }

    // - cutscenes section --------------------------------------------------

    // Stage the cutscene inputs into TempDir/game (the SA_ROOT tree the other sections share):
    //   ANIM/CUTS.IMG       - intro1a .cut/.dat (camera), .cut TEXT (subtitles), .ifp (ANPK anim)
    //   MODELS/PLAYER.IMG   - the csplay/CJ cutscene actor (PLATFORM-NEUTRAL -> bakes on PS2)
    //   MODELS/CUTSCENE.IMG - the cssmoke actor + csbat/csframe/csmomchair props (PS2-NATIVE VIF
    //                          geometry, flags 0x01010037 -> deferred to the ambient-ped codec,
    //                          task #36; the bakers skip them and write valid containers)
    //   TEXT/AMERICAN.GXT   - subtitle strings
    //   AUDIO/CONFIG/*.DAT + AUDIO/STREAMS/CUTSCENE.PAK - the audio attempt (PS2 stream is VAG,
    //                          not PC-OGG -> cutaudio soft-skips the .ogg; subtitles still bake)
    // Self-contained: reuses whatever the peds/audio sections already staged. stageId "" =>
    // always runs with the section (writes into TempDir, not data/).
    private static bool StepExtractCutsceneInputs(ConvertContext cx)
    {
        string gameRoot = Path.Combine(cx.TempDir, "game");
        int got = 0, reuse = 0, miss = 0;

        var wanted = new[]
        {
            "ANIM/CUTS.IMG",
            "MODELS/PLAYER.IMG",            // csplay/CJ (reused if the peds section staged it)
            "MODELS/CUTSCENE.IMG",
            "TEXT/AMERICAN.GXT",
            "AUDIO/STREAMS/CUTSCENE.PAK",
        };
        foreach (var p in wanted)
        {
            string dest = Path.Combine(gameRoot, p.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(dest)) { ++reuse; continue; }
            var e = cx.Iso!.Find(p);
            if (e is null) { cx.Log($"   miss {p}"); ++miss; continue; }
            if (e.Size > 32L * 1024 * 1024)
                cx.Log($"   extracting {p} ({e.Size / (1024 * 1024)} MB)...");
            cx.Iso.ExtractTo(e, dest);
            ++got;
        }

        // AUDIO/CONFIG/*.DAT (StrmPaks/TrakLkup for the track lookup). Reuse if the audio
        // section already staged them.
        foreach (var e in cx.Iso!.ListAll())
        {
            if (e.IsDirectory || !e.Path.StartsWith("AUDIO/CONFIG/")) continue;
            string dest = Path.Combine(gameRoot, e.Path.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(dest)) { ++reuse; continue; }
            cx.Iso.ExtractTo(e, dest); ++got;
        }

        cx.Log($"   cutscene inputs staged into {gameRoot} ({got} extracted, {reuse} reused, {miss} missing)");
        // CUTS.IMG feeds the reliable outputs (camera + subtitles) and the ANPK actor anim.
        return File.Exists(Path.Combine(gameRoot, "ANIM", "CUTS.IMG"));
    }

    // Bake the cutscene camera + actor models into <data>/cutscene. StepBakeWorld-style: SA_ROOT
    // points the patched bakers at the disc extract; each takes its explicit output path as
    // argv[1] (so it skips its dev-loop memstick mirror). The camera (codec-free CSV parse)
    // always produces intro1a_cam.bin; cutscene_bake writes cutscene.bin with the platform-neutral
    // csplay/CJ actor and SKIPS the PS2-native cssmoke; cutprops writes a valid (empty on PS2)
    // CPRP - its props are PS2-native too. cam + actors are fatal (they produce real output on a
    // valid disc and exit 0 after the internal native-skip); props is non-fatal (fully deferred).
    private static bool StepBakeCutscene(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - cutscene bake skipped (install Python 3 + Pillow)"); return true; }

        string saRoot = Path.Combine(cx.TempDir, "game");
        if (!File.Exists(Path.Combine(saRoot, "ANIM", "CUTS.IMG")))
        { cx.Log("   CUTS.IMG missing - extract step did not run"); return false; }

        string cutOut = Path.Combine(cx.OutDir, "cutscene");
        Directory.CreateDirectory(cutOut);
        var env = new Dictionary<string, string> { ["SA_ROOT"] = saRoot };

        // (script, args, fatal). Each baker takes its explicit output path as argv[1].
        var chain = new (string script, string[] args, bool fatal)[]
        {
            ("cutscene_cam_bake.py", new[] { Path.Combine(cutOut, "intro1a_cam.bin") }, true),
            ("cutscene_bake.py",     new[] { Path.Combine(cutOut, "cutscene.bin") },    true),
            ("cutprops_bake.py",     new[] { Path.Combine(cutOut, "cutprops.bin") },    false),
        };
        foreach (var (script, args, fatal) in chain)
        {
            string? sc = PythonRunner.FindScript(script);
            if (sc is null) { cx.Log($"   {script} not found - cutscene bake aborted"); return false; }
            cx.Log($"   -> {script} {string.Join(' ', args)}");
            if (!PythonRunner.Run(s_python, sc, args, cx.Log, env, null, cx.Ct, cx.OnPercent))
            {
                if (fatal) { cx.Log($"   {script} FAILED - cutscene bake aborted"); return false; }
                cx.Log($"   {script} FAILED - skipped (cutscene props are PS2-native VIF, codec #36 pending)");
            }
        }
        cx.Log($"   cutscene models + camera baked into {cutOut}");
        return true;
    }

    // Bake the cutscene audio + subtitles -> <data>/cutscene. NON-FATAL: cutaudio takes the OUTPUT
    // DIRECTORY as argv[1] and emits intro1a_subs.bin (codec-free, always) plus intro1a.ogg (only
    // when the stream is a PC XOR-OGG container; the PS2 disc's CUTSCENE stream is VAG, so the
    // audio soft-skips like ambience_bake - the cutscene plays with subtitles + camera, no voice
    // track this pass).
    private static bool StepBakeCutAudio(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - cutscene audio skipped"); return true; }

        string saRoot = Path.Combine(cx.TempDir, "game");
        if (!File.Exists(Path.Combine(saRoot, "ANIM", "CUTS.IMG")))
        { cx.Log("   CUTS.IMG missing - cutscene audio skipped"); return true; }

        string cutOut = Path.Combine(cx.OutDir, "cutscene");
        Directory.CreateDirectory(cutOut);
        var env = new Dictionary<string, string> { ["SA_ROOT"] = saRoot };

        string? sc = PythonRunner.FindScript("cutaudio_bake.py");
        if (sc is null) { cx.Log("   cutaudio_bake.py not found - skipped"); return true; }
        cx.Log($"   -> cutaudio_bake.py {cutOut}");
        if (!PythonRunner.Run(s_python, sc, new[] { cutOut }, cx.Log, env, null, cx.Ct, cx.OnPercent))
        { cx.Log("   cutaudio_bake.py returned non-zero - skipped (audio/subtitles deferred, non-fatal)"); return true; }
        cx.Log($"   cutscene audio + subtitles baked into {cutOut}");

        // The voice track is a stream, not part of CUTS.IMG, and carries no name on the
        // disc. Picking it by LENGTH was wrong: the CUTSCENE pack holds 141 elements
        // with closely spaced durations, and the nearest to the animation's 100.7 s was
        // a different scene. The subtitles cutaudio_bake just wrote say exactly when
        // someone speaks, so the baker matches the element that is loud in those windows
        // and quiet between them; length remains the fallback.
        StreamStep(cx,
                   new[] { Path.Combine(cx.TempDir, "_streams_scratch"),
                           "--intro", Path.Combine(cutOut, "intro1a.adp"), "100.7",
                           "--intro-subs", Path.Combine(cutOut, "intro1a_subs.bin") },
                   "cutscene voice", false);
        return true;
    }
}
