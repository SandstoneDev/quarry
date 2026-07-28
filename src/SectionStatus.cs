namespace Quarry;

// Analysis states describe an existing data/ folder; run states describe a live convert.
public enum SectionStatus
{
    NotBuilt,     // no manifest record / never baked
    NeedsUpdate,  // recorded with an older stage version
    UpToDate,     // recorded, version matches, outputs present + hashes match
    Corrupt,      // an output is present but size/hash mismatches
    Queued,       // scheduled this run, not started
    Running,      // baking now
    Done,         // finished this run
    Failed,       // errored this run
    Cancelled,    // stopped by the user
}

// Result of analyzing one section against an existing data/ folder.
public readonly record struct SectionAnalysis(SectionStatus Status, TimeSpan Estimate);

// A progress event emitted by the runner for one section.
public readonly record struct SectionEvent(string SectionId, SectionStatus State, int Percent, string? Line);
