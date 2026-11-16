# Methods

This document describes every processing step and the reasoning behind each choice.
Code references are given in brackets.

## 1. Data

Public dataset of 28 regular runners (Fukuchi, Fukuchi & Duarte 2017, *PeerJ* 5:e3298,
data DOI [10.6084/m9.figshare.4543435](https://doi.org/10.6084/m9.figshare.4543435), CC BY 4.0).
Each runner ran 30 s on an instrumented treadmill at 2.5, 3.5 and 4.5 m/s. Marker
trajectories were recorded at 150 Hz and the ground reaction force at 300 Hz. A standing
(static) trial with additional anatomical markers was recorded for each runner.

**Lab coordinate system** (checked from the marker data): X = anterior (running direction),
Y = vertical, Z = right. [`rungait/io.py`]

**Data-handling issues found and handled automatically:**

| Issue | Handling |
|---|---|
| Inconsistent file names (`RBDS001…` vs `RBDS03…`) | IDs normalised to three digits |
| Extra 3.0 m/s trials for two runners | Excluded (not part of the published protocol) |
| A marker absent from the whole file (MT1 in 7 trials) | Reconstructed from the other four foot markers (rigid body) |
| Swapped labels (R.MT1 ↔ R.MT5 in RBDS001, 2.5 and 3.5 m/s) | Detected from the rigid-body residual (46 mm → 1 mm) and corrected |
| Marker gaps (up to 36 frames) | Spline / rigid-body / fallback filling, see §2 |

## 2. Marker pre-processing [`rungait/qc.py`]

1. **Label-swap check.** For each rigid cluster (pelvis, thigh, shank, foot), the median
   rigid-body fit residual is computed. If it exceeds 10 mm, every pairwise label swap is
   tested; a swap is accepted only if it brings the residual below 10 mm and halves it.
2. **Short gaps (≤ 10 frames, 67 ms):** cubic spline.
3. **Longer gaps / absent markers:** rigid-body reconstruction from at least three visible
   markers of the same cluster (the missing marker's position in the static trial is
   transformed with the least-squares rigid transformation of the visible markers).
4. **Fallback:** if fewer than three cluster markers were visible, gaps ≤ 20 frames are
   bridged by spline. Anything still missing marks the frame as *invalid*; steps that
   contain invalid frames are excluded (none remained in this dataset).
5. **Filtering:** 4th-order zero-lag Butterworth low-pass, **18 Hz**. The cut-off was chosen
   with Winter's residual analysis on heel, forefoot, shank, thigh and pelvis markers of all
   84 trials (median 18 Hz, IQR 17–19.5 Hz; `results/residual_analysis.csv`). Re-running the
   whole pipeline with 12 Hz changed the angle validation results by ≤ 0.1° and the event
   timing SD by < 2 ms, so the results are not sensitive to this choice.

Every step is counted in `results/qc_all.csv` (46 177 filled marker samples out of 12.1
million, i.e. 0.38 %).

## 3. Segment kinematics [`rungait/kinematics.py`]

Calibrated anatomical systems technique (CAST):

* **Static calibration.** Anatomical frames are defined in the standing trial:
  * Pelvis: origin mid-ASIS; Z from left to right ASIS; X from mid-PSIS to mid-ASIS (orthogonalised).
  * Hip joint centre: Harrington et al. (2007) regression from pelvic width and depth.
  * Thigh: Y from knee joint centre (mid-epicondyles) to hip joint centre; Z from the epicondyles.
  * Shank: Y from ankle joint centre (mid-malleoli) to knee joint centre; Z from the malleoli.
  * Foot: X from heel to mid-metatarsal heads; Z from MT1 to MT5.
  All frames use the same convention as the lab: x anterior, y superior, z right.
* **Tracking.** In every frame, the rigid transformation that maps the static cluster onto
  the measured cluster is estimated by least squares using the SVD solution
  (Söderkvist & Wedin 1993; Kabsch). This is the same estimation problem as a 3D Helmert
  transformation without scale in geodesy and photogrammetry. The RMS residual of this fit
  is stored per frame and used as a quality measure (median: shank 0.3 mm, thigh 0.8 mm,
  foot 1.4 mm, pelvis 8 mm, where markers on the iliac crest move with soft tissue).
* **Joint angles.** Rotation of the distal relative to the proximal segment, decomposed as a
  Cardan Z-X-Y sequence (flexion/extension, ab/adduction, axial rotation), equivalent to the
  joint coordinate system of Grood & Suntay. Signs: hip flexion +, knee flexion +, ankle
  dorsiflexion +, adduction +, internal rotation + (left side mirrored).

## 4. Gait events [`rungait/events.py`]

* **Force plate (reference).** Vertical GRF filtered at 30 Hz, threshold 50 N. Both feet act
  on the same treadmill plate, so each contact is assigned to the foot whose centre is lower
  at mid-stance. Contacts shorter than 80 ms or longer than 600 ms are rejected. The left and
  right sides alternated without exception in all 84 trials.
* **Marker only (alternative).** IC is the first frame in which any foot marker is within
  5 mm of its own ground level (2nd percentile of its height in the trial). TO is the first
  frame after the forefoot has been loaded in which both metatarsal markers are > 10 mm above
  their ground level. Thresholds were tuned on a few trials and then validated on all
  7 168 contacts (see `results/VALIDATION.md`).

## 5. Metrics [`rungait/metrics.py`]

Per contact: contact time, flight time, step time, step frequency, step length
(= belt speed × step time), duty factor, vertical oscillation of the pelvis centre over the
stride, peak vertical GRF, vertical average loading rate, foot-strike angle, knee flexion at
contact and peak in stance, knee excursion, peak hip adduction in stance, peak ankle
dorsiflexion in stance, and horizontal distance of the ankle joint centre ahead of the
pelvis at contact ("overstride").

* **Loading rate:** mean slope of the vertical GRF between 20 % and 80 % of the impact peak.
  Without a distinct impact peak (common in fore-/midfoot strikers) the force at 13 % of
  stance is used as the reference; `impact_peak_present` records which definition was used.
* **Foot-strike angle:** sagittal angle of the foot's long axis at IC minus the same angle in
  the standing trial. Rearfoot > 8°, forefoot < −1.6°, midfoot in between
  (Altman & Davis 2012).
* **Asymmetry:** symmetry index SI = 100·(R − L)/mean(R, L) for ratio-scale metrics; for
  angles and positions (where zero is arbitrary) the difference R − L is reported instead.
  The report flags |SI| > 10 % and |R − L| > 5° / 5 cm. These thresholds are illustrative
  screening values, not clinical cut-offs.
* **Curves:** time-normalised to 101 points from IC to the next IC of the same foot
  (gait cycle) or from IC to TO (stance), mean ± SD over all valid steps.

## 6. Validation [`scripts/validate.py`]

1. Sagittal hip, knee and ankle angle curves against the Visual3D-processed curves
   published with the dataset (RMSE, offset, Pearson r).
2. Marker-only vs force-plate events (bias, SD, 95 % limits of agreement).
3. Peak vertical GRF against the published processed GRF.
4. Foot-strike classification against the strike pattern in the participant metadata.
5. Force plausibility: mean vertical GRF over the trial must equal body weight.

## 7. Group statistics [`scripts/speed_effects.py`]

Each runner contributes one value per speed (mean over all valid steps of both sides).
Speed effects are tested with the Friedman test (non-parametric repeated measures);
Kendall's W is reported as the effect size. 14 metrics are tested without correction for
multiple comparisons, so the analysis is exploratory.

## 8. PROMs [`rungait/proms.py`]

KOOS scoring (five subscales, 0–100, 100 = no problems) with the user's-guide rule for
missing items (≤ 2 missing per subscale are replaced by the subscale mean; otherwise the
subscale is not scored). Invalid entries (outside 0–4) raise an error instead of being
scored. The example data are synthetic.

## References

* Altman AR, Davis IS (2012). A kinematic method for footstrike pattern detection in barefoot and shod runners. *Gait & Posture* 35:298–300.
* Fukuchi RK, Fukuchi CA, Duarte M (2017). A public dataset of running biomechanics and the effects of running speed on lower extremity kinematics and kinetics. *PeerJ* 5:e3298.
* Grood ES, Suntay WJ (1983). A joint coordinate system for the clinical description of three-dimensional motions. *J Biomech Eng* 105:136–144.
* Harrington ME et al. (2007). Prediction of the hip joint centre in adults, children, and patients with cerebral palsy based on magnetic resonance imaging. *J Biomech* 40:595–602.
* Michaud B, Begon M (2021). ezc3d: An easy C3D file I/O cross-platform solution for C++, Python and MATLAB. *JOSS* 6:2911.
* Robinson RO, Herzog W, Nigg BM (1987). Use of force platform variables to quantify the effects of chiropractic manipulation on gait symmetry. *J Manipulative Physiol Ther* 10:172–176.
* Roos EM et al. (1998). Knee Injury and Osteoarthritis Outcome Score (KOOS) – development of a self-administered outcome measure. *J Orthop Sports Phys Ther* 28:88–96.
* Söderkvist I, Wedin PÅ (1993). Determining the movements of the skeleton using well-configured markers. *J Biomech* 26:1473–1477.
* Winter DA (2009). *Biomechanics and Motor Control of Human Movement*, 4th ed. Wiley.
