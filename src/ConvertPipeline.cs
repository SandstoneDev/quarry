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
                new Step("Write default settings", "settings", 1, StepWriteSettings),   // b982: a fresh install boots with logging OFF
            }),
        new("world", "World map (geometry, textures, collision, night, foliage, signs)",
            "The whole world map baked from your disc: geometry, native textures, " +
            "collision, night lighting, grass and road signs. The slow one (~1.5 h).",
            DefaultOn: true, Available: true, Steps: new[]
            {
                new Step("Extract world inputs", "",      0,   StepExtractWorldInputs),
                new Step("Bake world map",       "world", 780, StepBakeWorld),   // v780: tools/world_store_build.py + world_store_verify.py --ref wired into
                                                                     // the chain (after pmap_lattice_verify.py, before tile_pack.py) - dedups every
                                                                     // tile's model/texture blob bytes into one world.idx+world.dat and writes a second,
                                                                     // STRIPPED copy of the 184 tiles into world/ps2global, beside world/ps2full (measured
                                                                     // 374.76 -> 171.39 MiB, 54.3% saved). Opt-in only: world/chunkset.txt keeps defaulting
                                                                     // to "ps2full" - the engine's `wsstore` toggle also defaults off (not in DebugMenu.c's
                                                                     // DM_SHIPPING; no settings.txt on a fresh install), so a fresh install pointed at
                                                                     // ps2global with the toggle off would find every region tile refused (pmap_load rc=-3)
                                                                     // - see StepBakeWorld's own comment on the store block for the full reasoning. A
                                                                     // build or verify failure only discards ps2global itself, the ~1.5 h ps2full bake
                                                                     // ships regardless. Bumping forces a re-bake past the incremental manifest so
                                                                     // existing installs get it.
                                                                     // v779: pmap_lattice.py + pmap_lattice_verify.py wired into the chain (after
                                                                     // pmap_lz4.py, before tile_pack.py) - every model's vertices onto ONE shared 1/128
                                                                     // lattice, replacing 7203 distinct per-model quantisation grids that seamed at every
                                                                     // tile boundary. Non-fatal: the pass self-checks its own output and writes atomically
                                                                     // before it is allowed to touch disk (see the chain's own comment for why that makes
                                                                     // non-fatal safe here). Bumping forces a re-bake past the incremental manifest so
                                                                     // existing installs get it.
                                                                     // v778 (b952): tessellation REMOVED from the chain - it corrupted UVs and the night sidecars; see the
                                                                     // note at the chain. The per-tile archive pack stays.
                                                                     // v777 (b946): the chain ended with guard-band tessellation (18u) and
                                                                     // the per-tile archive pack. Bumping forces a re-bake so both reach existing installs.
                                                                     // v776: ps2dff trims a mesh to its BinMesh vertex count - padding decoded as geometry drew as spikes over Las Venturas. v774: the decal classifier now requires SPARSE ink, so baked shadows stop rendering as black patches (v771: ps2_uv_tess caps each triangle's UV extent). Bumping forces a re-bake past the incremental manifest
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
                new Step("Bake SFX pool",        "audio-sfx", 10, StepBakeSfx),   // v10 (b971): bank 59 keeps sounds 1 and 2, the radio retune static bed. radio_static_start() names those ids; without them baked the handles resolve to a silent 2-frame VAG and the retune is inaudible - the fourth time this project has shipped a subsystem whose bank was never baked. The keepset change alone does NOT reach the card: this version is what forces the re-bake past the incremental manifest. v9 (b915): bank 52 keeps sound 10, the flamethrower gas loop (PlayFlameThrowerIdleGasLoop: BankSlotID 5 = bank 52, SoundID 10). v8 (b906): bank 59 keeps the frontend PICKUP pairs 16/17 and 27/28 (CAEFrontendAudioEntity::AddAudioEvent) - weapon pickups were firing a Desert Eagle because the audio EVENT id was read as a weapon-bank sound id. v7 (b904): bank 143 GENRL_WEAPONS is now baked resident, trimmed to the 31 sound ids Fire.c names. It was never in the pack, so EVERY firearm in the port was silent - the same miss as bank 52 below and the radio's "0 stations". v6 (b836): bank 52 GENRL_EXPLOSIONS is now baked resident (sounds 1-4). It was never in the pack, so the port had no explosion sound at all; without this re-bake the new CExplosionAudio finds nothing to play. v793: BEATS is staged with the audio inputs now, so the mission jingle finally lands in sfx.bin; bumping forces the re-bake past the incremental manifest. v4 (b821-824): the pool became the sound ARENA - sfx_index.bin v3 carries the disc's distance curve and event volumes beside the bank records, and sfx.bin is gone. v5 (b827): index v4 adds the per-surface audio class, which is what lets a footstep pick its FEET bank
                new Step("Bake ambience zones",  "audio-amb",  2, StepBakeAmbience),   // the zone table. v2 (b823-824): ambzones.bin v3 carries the auzo BOXES and SPHERES with their names, which is what lets an outdoor zone be selected by position and a scripted zone be switched on at all
                new Step("Bake ambience tracks", "audio-ambx", 4, StepBakeAmbienceTracks), // the audio behind it. v2: the ADPCM frame grid starts 4 bytes after the element header - decoding from the old offset produced noise v4 (b830): the PS2 stream layout was wrong in EVERY period - header 0x1F84 not 0x2000+4, and a radio element's audio blocks sit at +0x1000 behind two 750 Hz sub-streams, not at +0 - so every .adp on disc has to be rebuilt
                new Step("Bake radio",           "audio-radio",5, StepBakeRadio),      // slow: ~1.5 GB of stations. v2: same 4-byte stream-data offset fix. v4: stage the boot elf so stations get their names v5 (b830): the PS2 stream layout was wrong in EVERY period - header 0x1F84 not 0x2000+4, and a radio element's audio blocks sit at +0x1000 behind two 750 Hz sub-streams, not at +0 - so every .adp on disc has to be rebuilt v5 (b830): the PS2 stream layout was wrong in EVERY period - header 0x1F84 not 0x2000+4, and a radio element's audio blocks sit at +0x1000 behind two 750 Hz sub-streams, not at +0 - so every .adp on disc has to be rebuilt
                new Step("Bake load tunes",      "audio-tune", 1, StepBakeLoadtune),   // non-fatal (only SFX aborts the section)
            }),
        new("vehicles", "Vehicles",
            "Cars, bikes and planes baked from your disc: PS2-native geometry, native textures, " +
            "carcols paint, the damage panels and the embedded collision spheres, plus each model's " +
            "handling. Produces the default car, the whole roster and the model index.",
            DefaultOn: true, Available: true, Steps: new[]
            {
                new Step("Extract vehicle inputs", "",         0, StepExtractVehicleInputs),
                new Step("Bake vehicle roster",    "vehicles", 7, StepBakeVehicles),   // v7 (b837): planes bake as PLN2 - u8 wheelParent[4] carries the DFF parenting, so the wheels ride the gear frames instead of vanishing. Also emits the Hydra's real nozzle frames. v6 (b834): the plane bake emits wheel_rm_dummy/wheel_lm_dummy as nozzle_r/nozzle_l - the Hydra's REAL VTOL nozzle frames (CPlane::PreRender 0x6FED50 nodes 3 and 6). b440 had guessed misc_a/misc_b, which are gear parts; the engine now drives those as gear, so without this re-bake the Hydra has no rotating nozzles. v5: same ps2dff overrun trim (rumpo, petro, marquis, skimmer, vcnmav carried stray vertices). car.bin + veh_index + veh/*.bin (per-vehicle non-fatal). v2: vehicle position scale fixed (was 8x). v4: the wheel mesh is picked by rule, so buccanee, intruder, petrotr, combine and raindanc stop baking with no wheels - bump forces a re-bake past the incremental manifest
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
                new Step("Bake cutscene audio",     "cutaudio",  6, StepBakeCutAudio),   // v2: the voice track now comes off the disc as ADPCM. v3: same 4-byte stream-data offset fix. v4: the take is chosen by subtitle timing, not by length (length picked the wrong scene) v6 (b830): the PS2 stream layout was wrong in EVERY period - header 0x1F84 not 0x2000+4, and a radio element's audio blocks sit at +0x1000 behind two 750 Hz sub-streams, not at +0 - so every .adp on disc has to be rebuilt v6 (b830): the PS2 stream layout was wrong in EVERY period - header 0x1F84 not 0x2000+4, and a radio element's audio blocks sit at +0x1000 behind two 750 Hz sub-streams, not at +0 - so every .adp on disc has to be rebuilt
            }),
        new("interiors", "Interiors",
            "Interior world (safehouses, shops, missions) baked from GTA_INT.IMG + the " +
            "interior IPL/IDE maps, plus the entry/exit door markers (enex).",
            DefaultOn: true, Available: true, Steps: new[]
            {
                new Step("Extract interior inputs", "",          0, StepExtractInteriorInputs),
                new Step("Bake enex markers",       "enex",      1, StepBakeEnex),
                new Step("Bake interiors",          "interiors", 3, StepBakeInteriors),   // v3 (b975): map_export/pack.py gained the DECAL alpha class on 2026-08-05 (b0ebf52) - floor dust, stains and cracks stopped being classified as opaque - but nothing bumped this step, so the incremental manifest kept serving interiors baked BEFORE it. Measured on the card: interior_CARLS.pmap dated Jul 29 carries 0 decal textures out of 114 (106 opaque, 5 cutout, 3 blend), which is why the dust plane on CJ's floor rendered as a solid black quad while the interior decal pass added for it drew nothing at all. Same trap as audio-sfx v10: the classifier change alone does NOT reach the card. // v2: the PS2 DFF codec kept only the first VIF batch of a multi-batch stream - interiors lost up to 88% of a room's geometry
            }),
        new("hud", "HUD, fonts & menus",
            "HUD sprites, fonts, menu backgrounds, radar and loading screens baked from " +
            "your disc's TXDs (HUD/FONTS/FRONTEN1/2, gta3.img radar tiles).",
            DefaultOn: true, Available: true, Steps: new[]
            {
                new Step("Extract HUD inputs", "",      0, StepExtractHudInputs),
                new Step("Bake HUD & fonts",   "hud",   5, StepBakeHud),   // v5: hud.bin HUD2 adds the aim reticles (sitem16, siterocket); v4: loading arts come from LOADS<region>.txd
                new Step("Bake radar",         "radar", 2, StepBakeRadar),   // v2: RDR6 - tiles at their native 128x128 with a palette each, instead of one atlas squeezed to a third of the resolution; bump forces the re-bake past the incremental manifest
                new Step("Bake save icon",     "saveicon", 1, StepBakeSaveIcon),   // non-fatal
            }),
        new("peds", "Player & peds (CJ, character, pedestrians)",
            "CJ and the pedestrian models + their animations from your disc. The playable " +
            "hero (skinned CJ + locomotion/idle/fight clips) is baked from PLAYER.IMG + PED.IFP.",
            DefaultOn: true, Available: true, Steps: new[]
            {
                new Step("Extract ped inputs", "",     0, StepExtractPedInputs),  // PLAYER.IMG + PED.IFP (+ gta3.img)
                new Step("Bake hero (CJ)",     "hero", 16, StepBakeHero),         // FATAL: no hero = no playable game (v16 (b909): the MELEE combo blocks - baseball/knife/sword/chainsaw/dildo/flowers and fight_b..e. Two gates had to open: the .ifp names in IFP_BLOCKS, and vehicle_block_clips(), which only emitted groups 88..117 and 11..32 and so dropped the melee range 33..45 even once merged. v15: + the FAT and MUSCULAR locomotion blocks; v14: FINGER tracks no longer dropped + the rocket/chainsaw carry gaits)
                new Step("Bake player char",   "char", 4, StepBakeChar),          // non-fatal
                new Step("Bake anim groups",   "anim", 1, StepBakeAnimGroups),    // non-fatal
                new Step("Bake melee combos", "melee", 1, StepBakeMelee),     // data/melee.dat -> melee.bin: the thirteen fight combos. Non-fatal: without it every melee weapon falls back to bare fists.
                new Step("Bake weapon table",  "weapon", 1, StepBakeWeapons),   // non-fatal (absent = no weapons)
                new Step("Bake weapon models", "wmdl", 3, StepBakeWeaponModels), // non-fatal (absent = no gun in hand / no HUD icon). v3: merge the extra atomics AT THEIR FRAME MATRIX (minigun2/sawbarl/petals) + bake the gunflash mesh
                new Step("Bake pickup icons", "pkup", 1, StepBakePickups),   // non-fatal (absent = weapons still lie in the world, the icon band does not)
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
        // The station names live in the boot executable, and nothing else on the pipeline
        // wants it, so it is staged HERE and named from SYSTEM.CNF rather than hardcoded.
        // It was never staged at all before, so File.Exists always failed, --elf was never
        // passed, and every station fell back to showing its two-letter pack code.
        var extra = new List<string> { Path.Combine(cx.OutDir, "audio", "radio") };
        string elfId = cx.Disc?.ElfId ?? "";
        if (elfId.Length > 0)
        {
            string elf = Path.Combine(cx.TempDir, "game", elfId);
            if (!File.Exists(elf))
            {
                var e = cx.Iso!.Find(elfId);
                if (e is not null) cx.Iso.ExtractTo(e, elf);
            }
            if (File.Exists(elf)) { extra.Add("--elf"); extra.Add(elf); }
            else cx.Log($"   {elfId} not on the disc - stations will show their pack codes");
        }
        return StreamStep(cx, extra.ToArray(), "radio", true);
    }

    private static bool StepBakeAmbienceTracks(ConvertContext cx) =>
        StreamStep(cx, new[] { Path.Combine(cx.TempDir, "_streams_scratch"),
                               "--no-stations",   // ambience only: this run used to overwrite
                                                  // radio.bin with zero stations, and only the
                                                  // step order kept the dial working
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

    // The ps2world bake chain. Exact run order, and which steps are FATAL vs
    // non-fatal, live in the `chain` array below - each entry's own comment
    // explains why it sits where it does; this comment stays high-level.
    //
    // Historical note: tools/ps2world_rebake.ps1 is a standalone dev script for
    // iterating on the early, load-bearing part of this chain (export through
    // pmap_lz4.py - col/lod/dyn/road/mflags/night all read the pre-compression v2
    // header, so lz4 cannot run any earlier than right after them) without a full
    // Quarry run. It predates tile_pack.py (b946) and the world-lattice pass, so it
    // is no longer a mirror of the live chain - useful as a faster loop for the
    // geometry/collision/UV steps, not as documentation of what actually ships.
    //
    // stageId "world" -> the version on the Step() declaration below drives the
    // incremental manifest: bump it whenever this chain's output changes, or an
    // unchanged re-convert skips this whole ~1.5 h bake and an existing install
    // never gets the fix.
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
        // The deduped alternative world - built and verified further down, only once
        // everything else below has finished writing worldDir. Not created here (only
        // world_store_build.py itself, or the failure-cleanup code beside it, owns this
        // directory's lifecycle) - see the store block's own comment for why.
        string storeDir = Path.Combine(cx.OutDir, "world", "ps2global");
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
            // World lattice: every model's vertices onto ONE shared 1/128-unit position grid
            // (tools/pmap_lattice.py), replacing the 7203 distinct per-model quantisation grids
            // the pilot bake produces - that grid mismatch is the seam down the middle of a
            // road you can see in daylight. It reads/writes the compressed model blobs directly,
            // hence AFTER pmap_lz4.py, same slot the tessellation pass below used to run in.
            //
            // Non-fatal, and safe to be non-fatal: pmap_lattice.py self-checks every file it
            // produces (verify_bytes(expect_lattice=True, ref=original), from the SAME verifier
            // run standalone below) before it is allowed to touch disk, and writes atomically
            // (temp file + os.replace) - a bad or interrupted tile is left on its OLD per-model
            // scale, not corrupted. A failure here degrades to "that tile keeps the seam it
            // already shipped with", never to a broken world.
            //
            // The tessellation failure below does NOT apply to this pass - worth spelling out,
            // since both run in this same slot and one of them already broke the world twice.
            // Tessellation ADDED vertices, and three things ride on vertex IDENTITY that it did
            // not carry along: .night (one u16 per vertex), .nightd (runs addressed by vertex
            // INDEX), and the header's vertex_bytes (what the engine divides to get the vertex
            // count). The lattice pass only ever REWRITES x/y/z of vertices that already exist --
            // same count, same order, same indices - so nothing that addresses a vertex by its
            // position in the pool can desync, because that position never moves. Proving that is
            // exactly what pmap_lattice_verify.py's count checks (and the self-check above) exist
            // for, not just assert.
            //
            // pmap_lattice_verify.py below runs WITHOUT --ref: there is no pre-pass copy of
            // worldDir sitting around at bake time to byte-compare against, so at chain time this
            // only re-runs the structural + lattice checks pmap_lattice.py's own self-check
            // already ran per file - a second read of the same claim, not the stronger byte-
            // exact proof a manual `--ref <pre-pass copy>` run gives (that is what validated this
            // pass against the real 184-tile world before it was wired in here: converted 14126,
            // refused 0, too_small 3, clean against a pre-lattice reference copy).
            //
            // Known acceptance gap: "refused 0" above was measured on a v2.01 disc. v1.03 and
            // v2.01 ship 3 world .dff models at different sizes (exclbr_hotl02_lvs, vgsespras01,
            // vgsn_polnb01 - all Las Venturas), so a v1.03 bake is not guaranteed to also read
            // refused 0. That is fine BY DESIGN: an oversized model refuses on its OWN, keeps its
            // old scale, and the rest of its tile still converts - but only because
            // pmap_lattice.py prints a non-zero refused count as a loud banner line, not one
            // number folded into a four-number summary (see its own main()). A v1.03 bake has
            // not been run to confirm this (~1.5 h, out of scope for the change that wired this
            // pass in) - if one ever reads refused > 0, that is the expected, contained gap, not
            // a new bug.
            ("pmap_lattice.py",       new[] { worldDir },                                false),
            ("pmap_lattice_verify.py",new[] { worldDir },                                false),
            // b946: guard-band tessellation. The GE performs NO X/Y clipping - a primitive with
            // any vertex outside 0 <= Xs,Ys < 4096 after the viewport transform is DISCARDED, and
            // the hardware divides only at the near plane (Sony GE Users Manual, section 10,
            // p58-60: "you must divide the primitives in advance"). With our viewport the largest
            // surviving triangle is the near plane times seven, so 18u pairs with a near of 2.6.
            // Runs on the v3 (LZ4) files, hence AFTER pmap_lz4.py. Non-fatal: an untessellated
            // world still plays, it just flickers geometry at the screen edge when you turn.
            // b952: TESSELLATION IS OUT OF THE CHAIN. It is correct about geometry - it cuts every
            // edge to the threshold and the split itself is crack-free - but a .pmap is not only
            // geometry, and three things ride on the vertex pool that it does not carry along:
            //   * UV is stored in an int16_t field that the GE reads UNSIGNED (pmap.h), so
            //     interpolating it as signed corrupts every U above 8.0 tiles -> stretched roads.
            //   * region_*.night is one u16 per vertex, aligned to the vertex pool, and
            //     region_*.nightd addresses glow runs by vertex INDEX. Adding vertices shifts both.
            //   * the header's vertex_bytes is what the engine divides to get the vertex count, and
            //     it was left at the pre-split value - so the night buffer was accepted and applied
            //     to the wrong vertices.
            // On hardware that showed up as stretched textures and broken baked night light, while
            // the holes it was meant to fix are already handled by the near plane (b945: the largest
            // surviving triangle is near * (2048/240) * tan(fov/2)). Cost was real too: +13% world
            // size and tile loads 59 -> 86ms median.
            // The tool stays in bakers/ and works; putting it back means teaching it the sidecars
            // and the header, and that belongs in the bake, not in a post-pass.
            // tile_pack.py is deliberately NOT here anymore (it used to be the last entry in
            // this array) - see the dedicated block below the loop, after the world store,
            // for why packing has to move to run AFTER that pass rather than simply last here.
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

        // - world store (tools/world_store_build.py) ---------------------------------------
        //
        // WHY HERE, exactly between the loop above and the tile-pack block below:
        // world_store_build.py reads worldDir's region_*.pmap AND copies every OTHER file it
        // finds there straight through as a "sidecar" (see its own module docstring's SIDECAR
        // FILES section) - so it must run only once nothing above is still going to write into
        // worldDir, which is everything through pmap_lattice_verify.py (pmap_lattice.py is the
        // last rewrite of the .pmap bytes themselves; the verify pass only reads).
        //
        // It must ALSO run BEFORE tile_pack.py: tile_pack does not touch .pmap bytes, but it
        // DOES add a new region_*.tile file into worldDir, and world_store_build's blanket
        // sidecar copy has no way to tell that file apart from a real one - pack worldDir first
        // and ps2global would ship a second, STALE, non-deduped copy of every tile bundled
        // inside a .tile archive that the engine's shipping default (tilearc=1, DebugMenu.c
        // DM_SHIPPING) would then open INSTEAD OF the stripped loose .pmap this pass just wrote,
        // silently defeating the entire store. Confirmed empirically while wiring this in (not
        // just reasoned about): packing worldDir first and THEN building the store from it does
        // leak exactly that stray .tile into ps2global's sidecar copy.
        //
        // FATAL-NESS, and why it differs from the plain log-and-skip above: ps2global is an
        // ADDITIONAL, opt-in world (see the chunkset.txt discussion below) - never the
        // load-bearing one - so neither a build nor a verify failure may abort StepBakeWorld
        // and throw away the ~1.5 h ps2full bake that already succeeded. But world_store_verify.
        // py --ref proves every blob in the store resolves BYTE-IDENTICAL to its source tile's
        // own bytes; a failure there means some blob is NOT that, which is actively dangerous to
        // ship (wrong geometry/textures that look like a rendering bug, not a load failure that
        // names itself - the same class of hazard pmap.c's build-stamp mismatch guards against
        // at load time). So, unlike every other non-fatal entry above, a failure here also
        // DELETES the partial or unverified world/ps2global, so nothing half-built is ever left
        // where a later, deliberate chunkset.txt edit could point at it.
        bool storeOk = false;
        {
            string? buildSc = PythonRunner.FindScript("world_store_build.py");
            if (buildSc is null)
                cx.Log("   world_store_build.py not found - world store skipped (non-fatal enhancement; ps2full still ships)");
            else
            {
                cx.Ct.ThrowIfCancellationRequested();
                // Peak disk, briefly: ps2full (~374.76 MiB, measured) plus the store this is
                // about to write (~171.39 MiB, measured) sit on disk AT THE SAME TIME - world_
                // store_build.py only ever READS worldDir, it never deletes or shrinks it (it
                // must stay the load-bearing world regardless of what happens below). Neither
                // side of this is cleaned up afterwards: ps2full because it always ships, and
                // ps2global because a failed/discarded store is the ONLY case this step deletes
                // anything, and a verified one is meant to stay (see ★C/★E below for why it is
                // not selected by default even so).
                cx.Log("   building the deduped world store (world/ps2global) alongside world/ps2full - both sit on disk at once, roughly +170 MiB peak for this step");
                cx.Log($"   -> world_store_build.py {worldDir} --out {storeDir} --force");
                bool built = PythonRunner.Run(s_python, buildSc,
                    new[] { worldDir, "--out", storeDir, "--force" }, cx.Log, env, null, cx.Ct, cx.OnPercent);
                cx.Ct.ThrowIfCancellationRequested();
                if (!built)
                    cx.Log("   world_store_build.py FAILED - skipped (non-fatal enhancement; ps2full still ships)");
                else
                {
                    string? verifySc = PythonRunner.FindScript("world_store_verify.py");
                    if (verifySc is null)
                        cx.Log("   world_store_verify.py not found - cannot verify the store it just built, discarding it (an unverified store must not ship)");
                    else
                    {
                        cx.Log($"   -> world_store_verify.py {storeDir} --ref {worldDir}");
                        bool verified = PythonRunner.Run(s_python, verifySc,
                            new[] { storeDir, "--ref", worldDir }, cx.Log, env, null, cx.Ct, cx.OnPercent);
                        cx.Ct.ThrowIfCancellationRequested();
                        if (verified)
                        {
                            storeOk = true;
                            cx.Log("   world store OK - world/ps2global ready (opt-in: point world/chunkset.txt at " +
                                  "'ps2global' AND enable the engine's `wsstore` debug-menu/settings.txt toggle - " +
                                  "either alone is inert, see the chunkset.txt comment below)");
                        }
                        else
                            cx.Log("   world_store_verify.py FAILED - the store does not verify, discarding it (ps2full still ships as the default world)");
                    }
                }
            }
            if (!storeOk && Directory.Exists(storeDir))
            {
                try { Directory.Delete(storeDir, recursive: true); }
                catch (Exception ex) { cx.Log($"   could not remove the failed world/ps2global: {ex.Message}"); }
            }
        }

        // - tile pack: LAST, for both worlds ------------------------------------------------
        //
        // b946: one archive per tile. A tile was up to 14 separate file opens and an open costs
        // ~34ms on a Memory Stick - 94% of a tile load was opening, not reading. Packs ps2full
        // exactly as before this change.
        //
        // ★C: also packs ps2global when the store above verified: world.dat only ever addressed the
        // BLOB bytes (one already-open handle for those - see world_store_build.py's own "why
        // the per-tile tables stay with the tiles" paragraph); it does nothing for the SIDECAR
        // files (.col/.lod/.dyn/.night/.nightd/.grass/.road/.mflags/...), and a stripped tile
        // still carries just as many of those as a full one - so the exact same up-to-14-opens
        // argument applies to ps2global unchanged, and packing it is cheap: world.idx/world.dat
        // are never matched by tile_pack's region_<rx>_<ry>.* glob, so they are left alone, one
        // pair of global files, beside 184 much SMALLER .tile archives (the blob bytes they used
        // to carry now live only in world.dat). Reasoned from the store's own measured breakdown
        // rather than measured on the real world here (a full conversion run was out of scope
        // for this change): idx 0.44 + dat 108.68 + stripped-tiles 5.67 = 114.79 of the 171.39
        // MiB store total, leaving ~56.6 MiB of sidecars to bundle with the 5.67 MiB of stripped
        // tiles - close to the ~61 MiB a packed ps2global measures, against ~374 MiB for the
        // equivalent ps2full pack. Confirmed mechanically (not just arithmetically) on a small
        // synthetic multi-tile fixture while wiring this in: tile_pack.py packs a world_store_
        // build.py output cleanly, world.idx/world.dat are left untouched by the glob, and the
        // packed store came out smaller than the packed full world on that fixture too.
        var packTargets = new List<(string dir, string label)> { (worldDir, "ps2full") };
        if (storeOk) packTargets.Add((storeDir, "ps2global"));
        foreach (var (dir, label) in packTargets)
        {
            cx.Ct.ThrowIfCancellationRequested();
            string? sc = PythonRunner.FindScript("tile_pack.py");
            if (sc is null) { cx.Log($"   tile_pack.py not found - {label} ships as loose files (non-fatal)"); continue; }
            cx.Log($"   -> tile_pack.py {dir}");
            if (!PythonRunner.Run(s_python, sc, new[] { dir }, cx.Log, env, null, cx.Ct, cx.OnPercent))
            {
                cx.Ct.ThrowIfCancellationRequested();
                cx.Log($"   tile_pack.py FAILED on {label} - it ships as loose files (non-fatal)");
            }
        }

        // chunkset selector for the engine (the pilot already wrote regions.bin).
        // Written only when absent, or when it already selects "ps2full" - this line
        // looks obviously correct in isolation (of course a fresh convert should select
        // the world it just baked), but it used to run unconditionally, EVERY convert,
        // even one triggered for something unrelated (HUD, audio, ...). An operator who
        // had deliberately repointed this file at a rollback directory (e.g.
        // "ps2full_pre_lattice") would silently lose that selection on the very next
        // unrelated run - with the world itself ALSO overwritten in place besides
        // (StepBakeWorld bakes into world/ps2full directly, no snapshot) - so the
        // documented rollback ("chunkset.txt + a World store toggle") did not actually
        // roll back to anything once that happened. A fresh convert (no chunkset.txt
        // yet) still gets "ps2full" as always; an existing, DIFFERENT selection is now
        // read as a deliberate operator choice and left alone.
        //
        // ★E: this still selects "ps2full" even when the store above verified OK. The engine
        // only ever admits ps2global's stripped (version 5) tiles when its OWN `wsstore` toggle
        // is on (DebugMenu.c / pmap.c's pmap_load) - and that toggle's shipping default is OFF
        // (it is not in DM_SHIPPING; a fresh install has no settings.txt to read one from
        // either). If this write named "ps2global" here, a fresh install would point straight at
        // a world every one of whose region tiles pmap_load refuses (rc=-3, "toggle off") - an
        // empty, unplayable map on the FIRST boot, with nothing in this convert's own log to
        // explain why once the operator is looking at the PSP instead of this console. Defaulting
        // to ps2full costs nothing: it is exactly as correct and complete as it always was, store
        // or no store. ps2global stays a real, verified, ready-to-use directory that an operator
        // opts into on purpose - by editing this file AND the engine's own toggle, the same two
        // switches the store has always been documented as needing (see the log line above) --
        // never something this convert flips on their behalf.
        string chunksetPath = Path.Combine(cx.OutDir, "world", "chunkset.txt");
        string? kept = WriteChunksetSelector(chunksetPath, "ps2full");
        if (kept is not null)
        {
            cx.Log($"   world/chunkset.txt already selects '{kept}' - keeping it (not overwritten by this convert)");
        }
        // night twin is an intermediate: its colours were folded into the .night
        // sidecars beside the day pmaps - drop it so it isn't shipped or tracked.
        try { if (Directory.Exists(nightDir)) Directory.Delete(nightDir, recursive: true); }
        catch { /* best-effort */ }

        cx.Log($"   world map baked into {worldDir}");
        if (storeOk)
            cx.Log($"   world store also baked into {storeDir} (opt-in - see above; not selected by default)");
        return true;
    }

    // Decides what StepBakeWorld's chunkset.txt write should actually do, as a pure
    // function of the file's CURRENT content - pulled out of StepBakeWorld so this
    // exact decision (keep an operator's selection vs (re)write the default) has a
    // test that does not need to run a ~40 min world bake to exercise it. Returns
    // null if it wrote `desired` (file was absent, or already said `desired`); returns
    // the file's existing content if that content named something ELSE and was left
    // untouched, so the caller can log what was kept.
    public static string? WriteChunksetSelector(string chunksetPath, string desired)
    {
        string? existing = File.Exists(chunksetPath) ? File.ReadAllText(chunksetPath).Trim() : null;
        if (existing is null || existing == desired)
        {
            File.WriteAllText(chunksetPath, desired);
            return null;
        }
        return existing;
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
                        p == "AUDIO/STREAMS/AMBIENCE.PAK" ||
                        // BEATS carries the mission-complete sting, which the SFX bake reads
                        // straight through as ADPCM. It used to be staged by the RADIO step,
                        // which runs later, so on a disc convert the SFX pass never found it
                        // and the mission end sound was silently missing from sfx.bin.
                        p == "AUDIO/STREAMS/BEATS.PAK" ||
                        // The two surface tables. audio_bake reads them straight off SA_ROOT/data
                        // (sa_audcurve.surface_classes) to give each footstep its audio class, and
                        // raises FileNotFoundError without them - same late-staging trap as the
                        // ELF and BEATS above, so they are pulled here rather than by a later step.
                        p == "DATA/SURFINFO.DAT" || p == "DATA/SURFAUD.DAT";
            if (!take) continue;
            string dest = Path.Combine(gameRoot, p.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(dest)) { ++reuse; continue; }   // e.g. AUDIOZON.IPL already staged by interiors
            if (e.Size > 32L * 1024 * 1024)
                cx.Log($"   extracting {p} ({e.Size / (1024 * 1024)} MB)...");
            cx.Iso.ExtractTo(e, dest);
            ++got;
        }
        // The PS2 executable, for exactly the reason BEATS is staged above: audio_bake reads
        // the vehicle audio table out of it and ABORTS without it (ElfNotFound), but the only
        // other place that stages it is the RADIO step, which runs AFTER the SFX bake. A
        // convert against a clean temp dir therefore failed the SFX pass every time; runs on
        // a warm temp only worked because some earlier convert had left the ELF behind.
        string elfId = cx.Disc?.ElfId ?? "";
        if (elfId.Length > 0)
        {
            string elf = Path.Combine(gameRoot, elfId);
            if (File.Exists(elf)) ++reuse;
            else
            {
                var ee = cx.Iso.Find(elfId);
                if (ee is not null) { cx.Iso.ExtractTo(ee, elf); ++got; }
                else cx.Log($"   {elfId} not on the disc - the SFX bake will abort");
            }
        }

        cx.Log($"   {got} audio input file(s) staged into {gameRoot} ({reuse} reused)");
        return got > 0 || reuse > 0;
    }

    // Bake the SFX pool -> the sound arena in data/audio/ (PRIMARY). PS2 disc bodies are already native
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
            // Gates only the PC fallback for the mission jingle, which decodes an OGG
            // through numpy + soundfile. A PS2 disc never reaches it: the BEATS stream is
            // already VAG and copies through on the stdlib, so bank 250 DOES ship from a
            // disc convert - the old "deferred to the radio pass" reading of this flag
            // has not been true since the PS2 path landed.
            ["QUARRY_SFX_NO_JINGLE"] = "1",
        };

        string? sc = PythonRunner.FindScript("audio_bake.py");
        if (sc is null) { cx.Log("   audio_bake.py not found - SFX bake aborted"); return false; }
        cx.Log($"   -> audio_bake.py {cx.OutDir}");
        if (!PythonRunner.Run(s_python, sc, new[] { cx.OutDir }, cx.Log, env, null, cx.Ct, cx.OnPercent))
        { cx.Log("   audio_bake.py FAILED - SFX bake aborted"); return false; }

        // The arena's three files, not the v1 pool. audio_bake stopped writing sfx.bin at
        // b820 and the engine stopped opening it, so checking for it aborted the whole SFX
        // section on every convert - while passing on a dev folder that still had the
        // stale file. sfx_banks.bin is checked too: without it the arena comes up with the
        // resident set only and every streamed engine bank goes quiet.
        string[] want = { "sfx_index.bin", "sfx_res.bin", "sfx_banks.bin", "vehaud.bin" };
        long total = 0;
        foreach (string n in want)
        {
            string f = Path.Combine(cx.OutDir, "audio", n);
            if (!File.Exists(f) || new FileInfo(f).Length < 64)
            { cx.Log($"   {n} missing/empty - SFX bake aborted"); return false; }
            total += new FileInfo(f).Length;
        }
        cx.Log($"   sound arena baked -> {Path.Combine(cx.OutDir, "audio")} ({total / 1024} KB across {want.Length} files)");
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
        // The zone table is the stdlib half and is written even when the track transcode
        // soft-skips, so its absence means the auzo parse itself failed - worth saying,
        // because the symptom in-game is no ambience anywhere and no other clue.
        string amb = Path.Combine(cx.OutDir, "audio", "amb", "ambzones.bin");
        if (!File.Exists(amb))
            cx.Log("   ambzones.bin was NOT written - every audio zone will be silent");
        else
            cx.Log($"   audio zones baked -> {amb} ({new FileInfo(amb).Length} bytes)");
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

    // data/settings.txt - the in-game settings the debug menu reads and writes. Like
    // boot.txt this is written only when ABSENT, so a re-convert never discards what the
    // player changed.
    //
    // Only the keys a fresh install should not inherit from the engine's own shipping table
    // are listed. The big one is `log`: session logging is a development instrument, it is
    // not free (it cost 42s of a 340s run once, at 1326 lines a frame), and a player has no
    // use for it. It stays switchable in WORLD/'Log recording' and via boot.txt log=1.
    //
    // `preset 2` = Balance, and it is deliberately LAST: dm_preset_load overwrites whole
    // groups of rows, so it has to run after the individual keys above or it would undo
    // them. Balance is what turns on the two adaptive controllers - Auto-DD (draw distance
    // closed-loop on frame work time) and, since b983, Auto ground LOD (drops the skyline
    // over 31ms, restores it under 26ms). It is already the engine's built-in default, so
    // this line does not change behaviour today; it is here so a future change to that
    // default cannot silently move what a converted install ships with.
    // groundlod / grass are no longer listed above: both are preset-managed rows, and
    // Balance sets them anyway.
    private const string DefaultSettings = """
        log 0
        lrauto 0
        logpin 0
        nearx10 20
        nearfloorx10 12
        tilearc 1
        preset 2
        """;

    private static bool StepWriteSettings(ConvertContext cx)
    {
        Directory.CreateDirectory(cx.OutDir);
        string path = Path.Combine(cx.OutDir, "settings.txt");
        if (File.Exists(path)) { cx.Log($"   settings.txt already present - left as it is ({path})"); return true; }
        var lines = DefaultSettings.Split('\n').Select(l => l.Trim()).Where(l => l.Length > 0);
        File.WriteAllText(path, string.Join("\n", lines) + "\n");
        cx.Log($"   wrote default settings (logging OFF) -> {path}");
        return true;
    }

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
        // --blocks also writes the streamed per-IFP-block clip files. SA loads animations
        // a block at a time and ref-counts them; the whole set does not fit resident (the
        // thirty vehicle groups alone name 128 clips across 21 blocks). Same bake, same
        // rig, so a streamed clip is byte-identical to a resident one.
        string blocksOut = Path.Combine(cx.OutDir, "anim", "blocks");
        Directory.CreateDirectory(blocksOut);
        cx.Log($"   -> hero_bake.py cj {dst} --blocks {blocksOut}");
        if (!PythonRunner.Run(s_python, sc, new[] { "cj", dst, "--blocks", blocksOut },
                              cx.Log, env, null, cx.Ct, cx.OnPercent))
        { cx.Log("   hero_bake.py FAILED - hero bake aborted"); return false; }

        if (!File.Exists(dst) || new FileInfo(dst).Length < 1024)
        { cx.Log("   hero.bin missing/empty - hero bake aborted"); return false; }
        cx.Log($"   hero (CJ) baked -> {dst} ({new FileInfo(dst).Length / 1024} KB)");
        return true;
    }

    // Bake the SA animation-group tables -> anim/groups.bin. This is what makes boarding a
    // Rustler different from boarding a Sentinel: handling.cfg gives every vehicle an anim
    // group, its '^' section defines thirty of those as a pair of AssocGroupIds plus
    // eighteen selector bits, animgrp.dat adds the walkcycle groups, and vehicles.ide maps
    // the model ids. The 118 hard-coded groups cannot come off the disc - the PS2 build
    // assembles that table at runtime into .bss - so they ship with the tool as
    // bakers/data/sa_anim_groups.json. NON-FATAL: without the file the engine falls back to
    // its generic CAR_ clips.
    private static bool StepBakeAnimGroups(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - anim groups skipped"); return true; }

        string gameRoot = Path.Combine(cx.TempDir, "game");
        foreach (var p in new[] { "DATA/HANDLING.CFG", "DATA/VEHICLES.IDE", "DATA/ANIMGRP.DAT" })
        {
            string dest = Path.Combine(gameRoot, p.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(dest)) continue;
            var e = cx.Iso!.Find(p);
            if (e is null) { cx.Log($"   miss {p} - anim groups skipped"); return true; }
            cx.Iso.ExtractTo(e, dest);
        }

        string? sc = PythonRunner.FindScript("anim_group_bake.py");
        if (sc is null) { cx.Log("   anim_group_bake.py not found - anim groups skipped"); return true; }

        string animOut = Path.Combine(cx.OutDir, "anim");
        Directory.CreateDirectory(animOut);
        var env = new Dictionary<string, string> { ["SA_ROOT"] = gameRoot };
        cx.Log($"   -> anim_group_bake.py --out {animOut}");
        if (!PythonRunner.Run(s_python, sc, new[] { "--out", animOut }, cx.Log, env, null, cx.Ct, cx.OnPercent))
        { cx.Log("   anim_group_bake.py failed - vehicles will use the generic clips"); return true; }

        string dst = Path.Combine(animOut, "groups.bin");
        if (!File.Exists(dst) || new FileInfo(dst).Length < 1024)
        { cx.Log("   groups.bin missing/empty - vehicles will use the generic clips"); return true; }
        cx.Log($"   anim groups baked -> {dst} ({new FileInfo(dst).Length / 1024} KB)");
        return true;
    }

    // Bake DATA/MELEE.DAT -> melee.bin, the thirteen hand-to-hand combos. NON-FATAL:
    // without it every melee weapon falls back to bare fists, which is what the port did
    // before this stage existed. The combo a weapon uses is named by CWeaponInfo, so this
    // table and weapon.bin have to come off the SAME disc.
    private static bool StepBakeMelee(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - melee combos skipped"); return true; }

        string gameRoot = Path.Combine(cx.TempDir, "game");
        string rel = Path.Combine("DATA", "MELEE.DAT");
        string dest = Path.Combine(gameRoot, rel);
        if (!File.Exists(dest))
        {
            var e = cx.Iso!.Find("DATA/MELEE.DAT");
            if (e is null) { cx.Log("   miss DATA/MELEE.DAT - melee combos skipped"); return true; }
            cx.Iso.ExtractTo(e, dest);
        }

        string? sc = PythonRunner.FindScript("melee_bake.py");
        if (sc is null) { cx.Log("   melee_bake.py not found - melee combos skipped"); return true; }

        var env = new Dictionary<string, string> { ["SA_ROOT"] = gameRoot };
        cx.Log($"   -> melee_bake.py --out {cx.OutDir}");
        if (!PythonRunner.Run(s_python, sc, new[] { "--out", cx.OutDir }, cx.Log, env, null, cx.Ct, cx.OnPercent))
        { cx.Log("   melee_bake.py failed - melee weapons will swing fists"); return true; }

        string dst = Path.Combine(cx.OutDir, "melee.bin");
        if (!File.Exists(dst)) { cx.Log("   melee.bin missing - melee weapons will swing fists"); return true; }
        cx.Log($"   melee combos baked -> {dst} ({new FileInfo(dst).Length} B)");
        return true;
    }

    // Bake DATA/WEAPON.DAT -> weapon.bin, the 80-record weapon table. NON-FATAL: without it
    // every weapon lookup returns nothing and the game stays unarmed, which is what it did
    // before this stage existed.
    private static bool StepBakeWeapons(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - weapon table skipped"); return true; }

        string gameRoot = Path.Combine(cx.TempDir, "game");
        {
            const string p = "DATA/WEAPON.DAT";
            string dest = Path.Combine(gameRoot, p.Replace('/', Path.DirectorySeparatorChar));
            if (!File.Exists(dest))
            {
                var e = cx.Iso!.Find(p);
                if (e is null) { cx.Log("   miss DATA/WEAPON.DAT - weapon table skipped"); return true; }
                cx.Iso.ExtractTo(e, dest);
            }
        }

        string? sc = PythonRunner.FindScript("weapon_bake.py");
        if (sc is null) { cx.Log("   weapon_bake.py not found - weapon table skipped"); return true; }

        var env = new Dictionary<string, string> { ["SA_ROOT"] = gameRoot };
        cx.Log($"   -> weapon_bake.py --out {cx.OutDir}");
        if (!PythonRunner.Run(s_python, sc, new[] { "--out", cx.OutDir }, cx.Log, env, null, cx.Ct, cx.OnPercent))
        { cx.Log("   weapon_bake.py failed - the game will be unarmed"); return true; }

        string dst = Path.Combine(cx.OutDir, "weapon.bin");
        if (!File.Exists(dst) || new FileInfo(dst).Length < 4096)
        { cx.Log("   weapon.bin missing/short - the game will be unarmed"); return true; }
        cx.Log($"   weapon table baked -> {dst} ({new FileInfo(dst).Length} B)");
        return true;
    }

    // Bake the weapon meshes -> weapons/w<type>.bin, one file per weapon so the runtime can
    // stream them. NON-FATAL: no mesh just means no gun in the hand and no HUD icon.
    // ★ There is no icon-texture step to add next to this one: models/hud.txd carries a
    // single weapon texture ("fist") and SA draws every other icon from the 3D model.
    private static bool StepBakeWeaponModels(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - weapon models skipped"); return true; }

        string gameRoot = Path.Combine(cx.TempDir, "game");
        {
            const string p = "DATA/DEFAULT.IDE";
            string dest = Path.Combine(gameRoot, p.Replace('/', Path.DirectorySeparatorChar));
            if (!File.Exists(dest))
            {
                var e = cx.Iso!.Find(p);
                if (e is null) { cx.Log("   miss DATA/DEFAULT.IDE - weapon models skipped"); return true; }
                cx.Iso.ExtractTo(e, dest);
            }
        }
        if (!File.Exists(Path.Combine(gameRoot, "MODELS", "GTA3.IMG")))
        { cx.Log("   MODELS/GTA3.IMG not staged - weapon models skipped"); return true; }

        string? sc = PythonRunner.FindScript("weapon_model_bake.py");
        if (sc is null) { cx.Log("   weapon_model_bake.py not found - weapon models skipped"); return true; }

        var env = new Dictionary<string, string> { ["SA_ROOT"] = gameRoot };
        cx.Log($"   -> weapon_model_bake.py --out {cx.OutDir}");
        if (!PythonRunner.Run(s_python, sc, new[] { "--out", cx.OutDir }, cx.Log, env, null, cx.Ct, cx.OnPercent))
        { cx.Log("   weapon_model_bake.py failed - no weapon meshes"); return true; }

        string wdir = Path.Combine(cx.OutDir, "weapons");
        int n = Directory.Exists(wdir) ? Directory.GetFiles(wdir, "w*.bin").Length : 0;
        cx.Log($"   weapon models baked -> {wdir} ({n} files)");
        return true;
    }

    // The pickup ICON band - dollar, bribe, info, health, armour and the rest. They are
    // declared in DATA/MAPS/GENERIC/DYNAMIC.IDE, NOT in DEFAULT.IDE, which is why a search
    // by name in the usual place comes up empty. Weapon pickups reuse the weapon meshes,
    // so this stage is only the icons. NON-FATAL: without it the weapons still lie in the
    // world and the icon pickups simply have no model.
    private static bool StepBakePickups(ConvertContext cx)
    {
        s_python ??= PythonRunner.FindPython();
        if (s_python is null) { cx.Log("   python not found - pickup icons skipped"); return true; }

        string gameRoot = Path.Combine(cx.TempDir, "game");
        foreach (var p in new[] { "DATA/MAPS/GENERIC/DYNAMIC.IDE", "DATA/MAPS/GENERIC/PROPEXT.IDE" })
        {
            string dest = Path.Combine(gameRoot, p.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(dest)) continue;
            var e = cx.Iso!.Find(p);
            if (e is null) { cx.Log($"   miss {p} - pickup icons skipped"); return true; }
            cx.Iso.ExtractTo(e, dest);
        }
        if (!File.Exists(Path.Combine(gameRoot, "MODELS", "GTA3.IMG")))
        { cx.Log("   MODELS/GTA3.IMG not staged - pickup icons skipped"); return true; }

        string? sc = PythonRunner.FindScript("pickup_bake.py");
        if (sc is null) { cx.Log("   pickup_bake.py not found - pickup icons skipped"); return true; }

        var env = new Dictionary<string, string> { ["SA_ROOT"] = gameRoot };
        cx.Log($"   -> pickup_bake.py --out {cx.OutDir}");
        if (!PythonRunner.Run(s_python, sc, new[] { "--out", cx.OutDir }, cx.Log, env, null, cx.Ct, cx.OnPercent))
        { cx.Log("   pickup_bake.py failed - no pickup icons"); return true; }

        string pdir = Path.Combine(cx.OutDir, "pickups");
        int n = Directory.Exists(pdir) ? Directory.GetFiles(pdir, "p*.bin").Length : 0;
        cx.Log($"   pickup icons baked -> {pdir} ({n} files)");
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
    // always produces intro1a_cam.bin.
    //
    // b96x, comment corrected: this block used to say cssmoke was SKIPPED as PS2-native and that
    // cutprops wrote an EMPTY CPRP on PS2. Both stopped being true and nobody updated the text.
    // What actually happens now:
    //   cutscene_bake bakes BOTH actors - ACTORS = [("cssmoke","index"), ("csplay","cjcut")].
    //   cssmoke is the PS2-native VIF skinned DFF out of cutscene.img, decoded through
    //   tools/ps2skin (wired in 79f7744; its positions are s16 6.10 and came out 8x too large
    //   until b0ebf52). csplay is the real cutscene CJ assembled by hero_bake from player.img.
    //   cutprops_bake writes THREE rigid props - csbat, csframe, csmomchair - each as mesh plus
    //   the Root track of KRT0 keyframes from intro1a.ifp, so the runtime samples Root at the
    //   cutscene phase and draws the prop in the same world frame as the skinned actors.
    // Verified against the runtime loader's own output rather than the source: hardware log
    // deploy_psp/hw_b959/session_b959_hw.log line 80 reads "CUTSCENE: 2 actor(s) loaded" and
    // line 82 "CUTPROPS: 3 props loaded".
    //
    // cam + actors are fatal (they produce real output on a valid disc); props is non-fatal.
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
    // when the stream is a PC XOR-OGG container).
    //
    // b96x, comment corrected: this used to end "the PS2 disc's CUTSCENE stream is VAG, so the
    // audio soft-skips - no voice track this pass". That was overtaken by this same step's own
    // version notes: v2 took the voice off the disc as ADPCM, and v6 (b830) rebuilt every .adp
    // after finding the PS2 stream layout wrong in every period (header 0x1F84, and a radio
    // element's blocks at +0x1000 behind two 750 Hz sub-streams). The device carries a real
    // intro1a.adp - 2,883,600 bytes on the current card - so the cutscene does have voice.
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
                           "--no-stations",     // voice only: do not re-copy ~1.5 GB of stations
                           "--intro", Path.Combine(cutOut, "intro1a.adp"), "100.7",
                           "--intro-subs", Path.Combine(cutOut, "intro1a_subs.bin") },
                   "cutscene voice", false);
        return true;
    }
}
