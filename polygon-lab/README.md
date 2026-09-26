# polygon-lab

Annotation-format sandbox, living inside the LAByrinth prototype alongside the
GroundingDINO + SAM2 pipeline. Deliberately **outside** `temporal-bone-lab` so nothing
here can be committed to that repo by accident.

## Contents

| File | What it is |
|---|---|
| `polygon-lab.html` | Standalone sandbox. Draw regions, watch the export formats update live. Open it by double-clicking; no server, no npm, no network. |

Planned to live here too: the twice-weekly export script that reads polygons out of the
app's database and pushes them into a local Label Studio instance.

## The two conversions that matter

Both are implemented and commented in `polygon-lab.html`.

**Coordinates.** LAByrinth stores points normalised `0..1` relative to the image
(`normPoint` in `annotation-workbench.tsx` clamps to that range). Label Studio works in
percentages `0..100` of `original_width` / `original_height`. So the conversion is a
multiply by 100, and the reverse is a divide.

```
POLYGON   points = [[x, y], [x, y], ...]      // 0..1
RECT      points = [x, y, w, h]               // 0..1, top-left first
```

**Predictions, not annotations.** Imported polygons belong under `predictions` in a Label
Studio task. A *prediction* is a starting suggestion a human refines; an *annotation* is a
finished human label. Import them as annotations and Label Studio treats the work as done,
which defeats the point — the instructor's rough outline exists to be tightened into a mask.

## Label vocabulary

`polygon-lab.html` carries a hardcoded structure list only so it runs standalone. In the
real pipeline that list comes from the `AnatomicalStructure` table, and the LS labeling
config XML is generated from it. The "LS labeling config" tab shows exactly what that
generated XML looks like.

Note that `CriticalStructure` is **not** a usable source for this list without a
canonicalisation pass: the seed data spells the facial nerve four different ways and the
tegmen at least six, because each step's author wrote the name freehand.

## Running Label Studio locally

```
docker run -d --name label-studio -p 8080:8080 heartexlabs/label-studio:latest
```

Then `localhost:8080`. Nothing hosted, nothing on Render, no shared credentials.
