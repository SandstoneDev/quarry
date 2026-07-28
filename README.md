# Quarry

**Quarry** is a Windows tool that builds the data for the
[Sandstone Engine](https://github.com/SandstoneDev/sandstone-engine) from a PS2
disc image you already own.

No game assets are included here. Quarry only reads the disc image *you* provide
and writes out a `data/` folder the engine loads.

---

## Getting it

Grab `Quarry-vN.zip` from the [**Releases**](../../releases) - it's `Quarry.exe`
plus an embedded Python and every library it needs, so the only thing you install
is the .NET runtime the app itself needs.

### Requirements

**[.NET Desktop Runtime 8, x64](https://dotnet.microsoft.com/download/dotnet/8.0)**
- pick *Desktop Runtime* under .NET 8. Not the SDK, and not the plain runtime,
which has no Windows Forms. Windows offers the download itself if you run Quarry
without it.

Nothing else: Python, numpy, Pillow and lz4 all ship inside the folder.

(Or build it yourself - see *Building* below.)

---

## Using it

- Run `Quarry.exe`
- Point it at your own extracted PS2 disc image
- Tick the sections you want (world, vehicles, peds, HUD, interiors, audio, cutscenes …)
- Run - it writes a `data/` folder

Then copy that `data/` folder next to the engine's `EBOOT.PBP` on your PSP
(`PSP/GAME/SANDSTONE/`). See the
[engine repo](https://github.com/SandstoneDev/sandstone-engine) for the install
steps.

---

## Building

Needs only the **.NET 8 SDK** (`dotnet`). The embedded Python and the wheels are
fetched once into `.cache/`.

```powershell
pwsh -File build.ps1 -Clean
```

Produces `dist/Quarry/` (the runnable folder) and `dist/Quarry-vdev.zip`.

- `src/` - the Quarry C# GUI + convert pipeline
- `bakers/` - the Python bakers that read the disc and emit each data file

---

## Credits

- **SandstoneDev** - converter and engine. Developed with the help of
 [Claude Code](https://www.anthropic.com/claude-code).
- Special thanks to **DenielX** for help with the engine's graphics work -
 [GitHub](https://github.com/DenielX/) ·
 [YouTube](https://www.youtube.com/@sp-pteam-indev6976)

---

## Disclaimer

Provided for interoperability, preservation, and homebrew use with a copy of the
source game you legally own. This project ships **no copyrighted game data** - it
operates only on a disc image you supply. Trademarks and game content belong to
their respective owners; this project is not affiliated with or endorsed by any
game publisher. Rightsholders may reach the maintainer at `sandstonedevpsp@gmail.com`.
