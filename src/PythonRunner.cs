// Runs the project's Python bakers. phase 1: system python (PATH or the py launcher)
// + baker scripts found next to the exe (bakers/) or in the repo dev tree.
// phase 4: the release bundle (built by tools/quarry/build_bundle.ps1) ships an
// embedded CPython at python/python.exe beside Quarry.exe; FindPython prefers it
// so the shipped tool runs with ZERO installed dependencies. Its python314._pth
// (stdlib + . + Lib\site-packages + ..\bakers, and deliberately NO `import site`)
// makes the interpreter resolve the bundled numpy/Pillow/lz4 and the vendored
// gvcslib/formats/core WITHOUT any PYTHONPATH or -s flag, and stays isolated from
// the user's %APPDATA% site-packages.
using System.Diagnostics;
using System.Threading;

namespace Quarry;

public static class PythonRunner
{
    public static string? FindPython()
    {
        // Shipping bundle: the embedded interpreter beside the exe wins outright --
        // it must never silently fall back to a (possibly absent or broken) system
        // Python on the end user's machine.
        string bundled = Path.Combine(AppContext.BaseDirectory, "python", "python.exe");
        if (File.Exists(bundled)) return bundled;

        foreach (var (exe, probe) in new[] { ("python", "--version"), ("py", "-3 --version") })
        {
            try
            {
                var p = Process.Start(new ProcessStartInfo(exe, probe)
                { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false });
                if (p is null) continue;
                p.WaitForExit(5000);
                if (p.ExitCode == 0) return exe;
            }
            catch { /* not installed under this name; try the next */ }
        }
        return null;
    }

    /// Locate a baker script: bakers/ beside the exe (shipping layout), else the
    /// repo tools/ (dev runs from tools/quarry/Quarry/bin/...).
    public static string? FindScript(string name)
    {
        string exeDir = AppContext.BaseDirectory;
        string[] candidates =
        {
            Path.Combine(exeDir, "bakers", name),
            Path.Combine(exeDir, "..", "..", "..", "..", "..", name),          // tools/ from bin/Debug/net8.0-windows
            Path.Combine(exeDir, "..", "..", "..", "..", "..", "..", "tools", name),
        };
        foreach (var c in candidates)
            if (File.Exists(c)) return Path.GetFullPath(c);
        return null;
    }

    /// Full implementation: runs script with args, streams stdout/stderr lines into
    /// log, optionally reports baker progress (parsed from "NN%" or "a/b" patterns
    /// in stdout) and can be cancelled (kills the whole process tree). True = exit 0.
    /// The console window is suppressed (CreateNoWindow) so a headless baker run
    /// never flashes an empty conhost on screen.
    public static bool Run(string python, string script, string[] args, Action<string> log,
                           IDictionary<string, string>? env, string? workDir,
                           CancellationToken ct, Action<int>? onPercent = null)
    {
        var psi = new ProcessStartInfo(python)
        {
            RedirectStandardOutput = true, RedirectStandardError = true,
            UseShellExecute = false, CreateNoWindow = true,
            StandardOutputEncoding = System.Text.Encoding.UTF8,
        };
        if (python == "py") psi.ArgumentList.Add("-3");
        psi.ArgumentList.Add(script);
        foreach (var a in args) psi.ArgumentList.Add(a);
        psi.Environment["PYTHONUNBUFFERED"] = "1";   // stream baker stdout live (else CPython block-buffers a pipe -> the log looks frozen then dumps)
        if (env is not null) foreach (var (k, v) in env) psi.Environment[k] = v;
        if (workDir is not null) psi.WorkingDirectory = workDir;

        using var p = new Process { StartInfo = psi };
        void Handle(string? line)
        {
            if (line is null) return;
            log("   | " + line);
            if (onPercent is not null)
            {
                var m = System.Text.RegularExpressions.Regex.Match(line, @"(\d{1,3})\s*%");
                if (m.Success && int.TryParse(m.Groups[1].Value, out int pct)) onPercent(Math.Clamp(pct, 0, 100));
                else
                {
                    var f = System.Text.RegularExpressions.Regex.Match(line, @"\b(\d+)\s*/\s*(\d+)\b");
                    if (f.Success && int.TryParse(f.Groups[1].Value, out int a) && int.TryParse(f.Groups[2].Value, out int b) && b > 0)
                        onPercent(Math.Clamp(a * 100 / b, 0, 100));
                }
            }
        }
        p.OutputDataReceived += (_, e) => Handle(e.Data);
        p.ErrorDataReceived += (_, e) => { if (e.Data is not null) log("   ! " + e.Data); };
        p.Start();
        p.BeginOutputReadLine();
        p.BeginErrorReadLine();
        using (ct.Register(() => { try { if (!p.HasExited) p.Kill(entireProcessTree: true); } catch { } }))
            p.WaitForExit();
        return p.ExitCode == 0;
    }

    /// Run script with args; stream stdout/stderr lines into log. True = exit 0.
    public static bool Run(string python, string script, string[] args, Action<string> log) =>
        Run(python, script, args, log, null, null, CancellationToken.None);

    /// As above, plus per-run environment overrides (the world chain needs
    /// SA_ROOT / SA_GTA3_IMG pointing at the extracted disc) and an optional
    /// working directory.
    public static bool Run(string python, string script, string[] args, Action<string> log,
                           IDictionary<string, string>? env, string? workDir = null) =>
        Run(python, script, args, log, env, workDir, CancellationToken.None);
}
