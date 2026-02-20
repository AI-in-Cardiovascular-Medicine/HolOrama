# Contour Format Migration Checklist

## New format
```json
{
  "lumen": {
    "150": {
      "contours": [[[x1, x2, ...], [y1, y2, ...]]],
      "measurements": { "area": 1.23, "elliptic_ratio": 0.95, ... }
    }
  }
}
```
Keys that **stay top-level** (not contour-specific):
`phases`, `measures`, `measure_lengths`, `reference`, `angles`, `gating_signal`,
`plaque_frames`, `percent_stenosis_text`

---

## src/input_output/contours_io.py

- [ ] **L25–32** — Backward-compat checks add missing top-level keys (`measures`, `reference`, `angles`);
  need equivalent checks for measurement keys now inside contour dicts
- [ ] **L39** — `data['lumen'] = map_to_list(...)` assigns old tuple; XML read path must build new dict format
- [ ] **L40–56** — Initializes flat measurement keys (`lumen_area`, `elliptic_ratio`, etc.) per-frame;
  must move into `data['lumen'][frame]['measurements']` structure (for XML legacy path)
- [ ] **L62–65** — `set_data(main_window.data['lumen'], ...)` passes contour data; signature/call must adapt to new dict
- [ ] **L113–128** — XML write path accesses `data.get("lumen", [[], []])` then `lumen_data[0]`/`[1]`;
  must extract x/y from new `data['lumen'][frame]['contours'][0]` structure

---

## src/input_output/read_image.py

- [ ] **L78–90** — Initializes flat measurement keys as `[0] * num_frames`
  (`lumen_area`, `lumen_circumf`, `longest_distance`, `shortest_distance`, `elliptic_ratio`,
  `vector_length`, `vector_angle`, `eem_area`, `percent_stenosis_text`);
  most of these move inside contour dicts — only `percent_stenosis_text` stays top-level
- [ ] **L92–96** — Initializes `lumen`, `eem`, `calcium`, `branch` as `([...], [...])` tuples;
  must become `{}` dicts with per-frame keys
- [ ] **L102** — `set_data(main_window.data['lumen'], main_window.images)` passes old tuple; must adapt

---

## src/gui/left_half/IVUS_display.py

- [ ] **L202** — `data[ContourType.LUMEN.value] = lumen` assigns old tuple into data; must adapt to dict format
- [ ] **L225–246** — `_init_main_window_data()` creates `[[], []]` indexed tuple structure;
  must instead ensure an empty `{}` dict exists for each contour type key
- [ ] **L247–252** — `_start` / `_end` point keys are fine (top-level lists) — no change needed
- [ ] **L358–359** — `get_full_contour_list()` reads `data[key][0]` and `data[key][1]`;
  must iterate over `data[key].items()` sorted by frame number instead
- [ ] **L402–403** — `_get_contour_data()` returns `(data[0][self.frame], data[1][self.frame])`;
  must become `data[str(self.frame)]['contours'][0]` (first contour of that frame)

---

## src/gui/utils/contours_gui.py

- [ ] **L12–13** — Reads `data[key][0][frame]` and `[1][frame]` for edit buffer;
  must become `data[key].get(str(frame), {}).get('contours', [[]])[0]` etc.
- [ ] **L14** — Stores `tmp_contours[key] = (xlist, ylist)` — internal format OK if `_get_contour_data` is fixed

---

## src/gui/shortcuts.py

- [ ] **L148–149** — `remove_contours()` clears `data[key][0][frame]` and `[1][frame]`;
  must pop/clear `data[key].pop(str(frame), None)` or set empty contours inside the frame dict
- [ ] **L291–308** — `delete_contour()` accesses `contour_data[0][frame]` and `[1][frame]` for undo buffer;
  must read from `data[key].get(str(frame), {}).get('contours', [[]])[0]`
- [ ] **L323–328** — `undo_delete()` restores `contour_data[0][frame]` and `[1][frame]`;
  must write back into `data[key][str(frame)]['contours'][0]`

---

## src/report/report.py

- [ ] **L30** — `data['lumen'][0][frame]` detects contoured frames;
  must become `str(frame) in data['lumen']`
- [ ] **L99–116** — `compute_all()` reads flat top-level keys:
  `longest_distance`, `farthest_point[0/1]`, `shortest_distance`, `nearest_point[0/1]`,
  `lumen_area`, `lumen_circumf`, `lumen_centroid[0/1]`, `elliptic_ratio`, `vector_length`, `vector_angle`;
  all must be read from `data['lumen'][str(frame)]['measurements']`
- [ ] **L162–163** — Defensive init of `data['eem_area']` flat list; moves into EEM measurements dict
- [ ] **L169–171, 197** — Writes `data['eem_area'][frame]`; must write to `data['eem'][str(frame)]['measurements']['area']`
- [ ] **L181** — `lumen_area[frame] = ...` etc. (local variable write); follow-through from L99 fix — OK if references updated
- [ ] **L216–227** — DataFrame built from local vars derived from old flat keys; follows from L99 fix
- [ ] **L229–231** — Writes `data['elliptic_ratio']`, `vector_length`, `vector_angle` back as flat lists;
  must write back into each frame's measurements dict instead
- [ ] **L321–324** — `compute_polygon_metrics()` writes `data['lumen_area'][frame]`, `data['lumen_circumf'][frame]`,
  `data['lumen_centroid'][0][frame]`, `[1][frame]`;
  must write to `data['lumen'][str(frame)]['measurements']`
- [ ] **L373–375** — `farthest_points()` writes `data['longest_distance'][frame]`, `data['farthest_point'][0/1][frame]`;
  must write to `data['lumen'][str(frame)]['measurements']`
- [ ] **L414–416** — `closest_points()` writes `data['shortest_distance'][frame]`, `data['nearest_point'][0/1][frame]`;
  must write to `data['lumen'][str(frame)]['measurements']`

---

## src/segmentation/segment.py

- [ ] **L44** — `lumen = main_window.data['lumen']` reads as old tuple;
  must adapt to work with new dict (or re-assign into it frame by frame)
- [ ] **L54–58** — Writes `lumen[0][frame]` and `[1][frame]`;
  must write to `data['lumen'][str(frame)]['contours'] = [[x_list, y_list]]`
- [ ] **L24–25** *(commented out)* — `data['lumen_area'] = [0] * num_frames` reset; keep in mind when uncommenting
