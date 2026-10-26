# Automatic IVUS gating: method and validation

Working notes for the paper on the image-based gating in HolOrama. Everything here can be
rerun with the scripts in `scripts/` (see [Rerunning](#rerunning)). Status: 2026-10-26,
algorithm in `src/signal_processing/automatic_gating.py` after commit `461f966`.

## Data

- **Cohort**: NARCO IVUS pullbacks, Boston Scientific AVVIGO, 30 frames/s, resting
  acquisitions (Run1 or Rest folder), plus one stress pullback (`test_cases/anonymized.dcm`,
  about 156 BPM).
- **Reference labels**: diastole (D) and systole (S) marked manually per beat by A. W. Stark
  within one gated section per pullback. D is the stillest frame in diastole, S the stillest
  frame around end-systole. No ECG was available.
- **Groups** (`cases.csv`):
  - `tuning`: 10 pullbacks re-labelled and verified on 2026-10-26. Five are rated *perfect*
    (signal exactly as expected: 348, 359, 365, 367, 368), all ten are analysable.
    Perfect cases weigh double in the overall scores.
  - `earlier`: 12 pullbacks with the labels as of 2026-10-23, not re-verified. Corrections:
    D/S swapped in 329 and 330. Excluded: 323 (very bad signal) and 360 (labels and signal
    repeat every 13 frames, 138 BPM at 30 fps, while about 85 BPM was recorded).
  - `stress`: the anonymised test case. Its corrected labels are in `labels/test.csv`
    (the phases in the test-case JSON were later overwritten by auto-gating).

## Method (as implemented)

1. **Image motion signal**: `s[n] = 1 - NCC(frame_n, frame_n+1)` on the central 400 x 400
   crop, each frame scaled to [0, 1], then z-scored in windows of 100 frames. Low values
   mean consecutive frames are alike, i.e. the heart is still.
2. **Heart rate**: harmonic sum over the spectrum of `s`. For each candidate frequency f in
   45-200 BPM the spectral peaks at f, 2f, 3f, 4f are summed with weights 1, 0.9, 0.8, 0.7
   (peak within max(0.08 Hz, one bin) of each harmonic). The plain FFT peak locks onto the
   2nd or 3rd harmonic because `s` has two motion events per beat. The slight weight decay
   keeps a candidate at half the true rate from tying with it.
3. **Gating signal**: zero-phase 4th-order Butterworth bandpass at 0.7-2.2 x HR, z-scored
   in windows of 100 frames (the green curve in the app).
4. **Diastole, beat tracker**: candidates are all local minima of the gating signal.
   Dynamic programming over pairs of consecutive diastoles picks the chain maximising
   - node score: robust z-score of the valley depth (median / IQR), deeper is better,
   - regular step (0.7-1.35 beats): cost `5 (gap - 1)^2`,
   - tempo change between consecutive regular steps: cost `10 (gap_k - gap_k-1)^2`,
   - skipped beat (step of 1.55-2.3 beats, extrasystole or missed beat): cost 6.
   The chain starts and ends within 1.5 beats of the first and last candidate. Skipped
   intervals are stored and shaded in the gating plot for review.
5. **Lumen minor-axis clock** (only when contours cover at least 50 % of frames): phase of
   the minor axis bandpassed at 0.7-1.3 x HR (Hilbert transform). Its mean phase at the
   first-pass diastoles gives the patient-specific offset. A second pass adds
   `R cos(phase - offset)` to the node score, where R is the resultant length at those
   diastoles.
6. **Systole**: per regular beat, search window from `a + m` to `b - m` with
   `m = min(0.35 beat, 0.45 (b - a))`. The median delay from D to the deepest valley in the
   window is learned per pullback, then each beat takes the valley minimising
   `g + 64 ((t - a - delay) / beat)^2`. Skipped beats get no systole.
7. **Frequency sweep**: clicking a cutoff C in the sweep heat map sets HR = C / 2 and uses
   the signal low-pass filtered at C as gating signal. While the gating is unchanged since
   the last automatic run it is redone, a manually edited gating is only overwritten after
   confirmation.

Runtime of steps 4-6: at most 8 ms per pullback (up to 1700 frames).

## Evaluation protocol

- Only labelled sections are scored (first to last label of a pullback).
- **Correct**: a prediction of the same phase within 1 (or 2) frames of the label.
  **Swapped**: none of the same phase but one of the other phase within the tolerance.
- **Violations** (non-physiological output): consecutive diastoles closer than 0.7 beat,
  or a diastole and a systole closer than 0.25 beat.
- Overall totals weigh perfect cases double. Group totals are unweighted.
- `scripts/evaluate.py` runs the app's own `AutomaticGating` (no reimplementation).

## Results

### Current algorithm (2026-10-26, `results/evaluation_2026-10-26.txt`)

| Group | Variant | D correct / swapped (+-1) | S correct / swapped (+-1) | D / S correct (+-2) | Violations | Skipped |
|---|---|---|---|---|---|---|
| tuning (10) | image only | 78.9 / 9.4 % | 68.6 / 11.2 % | 84.2 / 80.2 % | 0 | 8 |
| tuning (10) | + minor axis | 79.9 / 8.8 % | 69.0 / 10.6 % | 84.8 / 80.9 % | 0 | 8 |
| stress (1) | + minor axis | 100 / 0 % | 97.2 / 0 % | 100 / 97.2 % | 0 | 0 |
| earlier (12) | + minor axis | 50.1 / 11.7 % | 50.0 / 3.8 % | 62.5 / 64.7 % | 0 | 4 |

Per-case results are in the matching `.csv`. In the earlier group a few pullbacks score
very low (366: 0 % D, 335: 9 %, 381: 18 %), more likely label problems than gating
errors. They are the first candidates for re-labelling.

### Development history (same scoring)

Tuning group, perfect cases double:

| Version | D correct (+-1) | S correct (+-1) | Violations |
|---|---|---|---|
| Two-step gating, commit 461f966 | 57.9 % | 61.6 % | 117 |
| Beat tracker, initial parameters | 72.5 % | 62.6 % | 0 |
| Beat tracker, tuned (image only) | 78.1 % | 69.9 % | 0 |
| Beat tracker, tuned + minor-axis clock | 78.9 % | 70.1 % | 0 |

Earlier labels, 21 pullbacks (2026-10-23 state, 351 restricted to frames 500-900,
including the pullbacks later re-labelled):

| Version | D correct (+-1) | S correct (+-1) |
|---|---|---|
| Original AIVUS-CAA style (hysteresis extrema + lumen-area classification) | 43.0 % | 28.2 % |
| Two-step gating (461f966) | 47.5 % | 44.8 % |
| Beat tracker, initial parameters | 57.9 % | 49.9 % |
| Beat tracker, tuned parameters | 58.1 % | 52.6 % |

The tuned parameters were chosen on the tuning group only. That they also improve the
earlier set (not used for tuning) argues against overfitting.

### Heart rate detection (24 pullbacks)

| Method | Within 10 % of the label-derived rate |
|---|---|
| FFT peak (45-200 BPM) | 15 / 24 (9 resting scans at 2-3 x the true rate) |
| Harmonic sum, 4 harmonics, weights 1 / 0.9 / 0.8 / 0.7 | 23 / 24 |

### Which features carry phase information (`results/analysis_*`)

- **Valley depth of the image signal separates D from S**: D is the deeper valley in a
  median 82 % of beats (18 of 22 pullbacks above 60 %). In the 10 re-labelled pullbacks it
  holds in all of them (63-100 %, median 88 %).
- **Image signal phase**: D sits near the minimum of the heart-rate component of `s` in
  every pullback (across-case resultant length of the D phase 0.70 over 22 pullbacks,
  within-case 0.67), S half a beat later.
- **Lumen minor axis**: strongly periodic at the heart rate (50-58 % of its spectral power
  in the heart-rate band in three checked pullbacks, versus 9-15 % for `s`), and D and S
  sit half a beat apart on it. But the phase at which diastole falls differs between
  patients (across-case resultant length 0.05, within-case 0.41). It can therefore serve
  as a beat clock with a learned offset, not as a stand-alone phase detector. Area
  (0.08), major axis (0.06), elliptic ratio (0.17) and blur (0.14) behave the same.

### Tried and rejected

| Idea | Result (tuning group, perfect x2) |
|---|---|
| Talou et al. 2016 (CCB): harmonic minima refinement on `s` | D 44.8 %, S 57.4 %, 149 violations |
| CCB with blur feature `-sum |grad u|`, alpha per pullback by minimal period SD | D 27.6 %, S 41.7 % |
| Tracker on the CCB combined signal | D 44.1 %, S 33.0 % |
| Tracker + harmonic refinement of the diastoles | D 47.3 %, S 65.0 % |
| Blur as D/S discriminator | direction flips between pullbacks (D sharper in 4 of 10) |
| Similarity to the frames one beat away (diastolic frames should look alike) | no gain at any weight, best D 78.1 %, S 69.9 % |
| Symmetric stillness `s[n-1] + s[n]` | D 74.0 %, S 68.8 % (labels follow the plain `s`) |
| Final shift to the raw `s` minimum within +-1 frame | D -5.8 points (labels sit on the filtered valleys) |
| Valley width as D/S discriminator (diastole lasts longer) | direction flips between pullbacks |
| Stronger rhythm penalties to hold the phase | lower D and S (heart rate drifts within a pullback) |
| Fixed image-phase prior added to the tracker | -0.7 to +0.2 points |

## Limitations

- **Phase ambiguity at high resting heart rates**: in 367 (103 BPM) frames 284-906 the
  labelled diastole is the shallower valley in 24 of 36 beats. The tracker follows depth
  there (D 51 %). The previous two-step version scored 56 % on this pullback.
- **Extrasystoles**: the bandpass can produce a ghost valley inside a compensatory pause.
  The tracker then places a diastole in the pause instead of flagging a skipped beat. The
  premature beat itself is never chosen. Skips are only flagged when no plausible valley
  exists.
- **Tuning set**: 10 pullbacks from one cohort, labelled by one observer. No inter-observer
  data. Two further parameters showed small gains on the tuning group only and were not
  adopted: systole margin 0.4 beat (S +2.2 points), delay weight 256 (S +0.2).
- **Temporal resolution**: 30 frames/s, so one frame is 33 ms.

## Rerunning

Paths: `NARCO_ROOT` (default `E:\PostDoc_Anselm-Stark\06_ivus_data\01_narco_ivus_raw`) and
`GATING_CACHE` (default `cache/` here, git-ignored, about 20 MB per pullback with frames).

1. Add the new pullbacks to `cases.csv` (group, folder `Run1`/`Rest`, perfect, quality).
   Set `swap_labels` or `label_frame_min/max` only if labels must be corrected.
2. Build their cache from the DICOM and the newest contour JSON:
   `python scripts/build_cache.py 401 402 ...` (add `--frames --blur` for the side
   analyses, `--rebuild` after re-labelling).
3. Score the shipped algorithm: `python scripts/evaluate.py` (or `--group tuning`).
4. Optional: `python scripts/tune.py --group tuning` for the parameter sweep and
   `python scripts/analyses.py phase|discriminate` for the feature analyses.

Run the scripts from `scripts/` with the project environment (they import `src`).

## Files

- `cases.csv`: case list, groups, ratings, label corrections, exclusions.
- `labels/`: label files overriding the JSON phases (`frame,phase`).
- `scripts/common.py`: paths, case loading, scoring.
- `scripts/build_cache.py`, `evaluate.py`, `tune.py`, `analyses.py`: see above.
- `results/`: dated outputs of the scripts.

## References

- Maso Talou G. D. et al. Improving cardiac phase extraction in IVUS studies by
  integration of gating methods. IEEE Trans Biomed Eng (TBME-00327-2015).
- Stark A. W. et al. (2025). Automated intravascular ultrasound image processing and
  quantification of coronary artery anomalies: the HolOrama software. Comput Methods
  Programs Biomed, 109065.
