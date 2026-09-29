# dancing

Extracting dance motion from video, fitting it to a closed-form curve, and
asking what the curve says about the choreography.

**[Live page →](https://raggedr.github.io/dancing/)** — the reconstruction runs
in your browser, evaluating the series below 50 times a second. Switchable
between the raw landmarks and the fitted curve.

## What this does

1. **Extract** — 2D pose landmarks and a person segmentation mask from any video,
   in one inference pass (`extract_motion.py`). Optional metric 3D landmarks
   (`extract_world.py`).
2. **Fit** — each joint coordinate becomes a truncated DCT-II series:

   ```
   x(t) = Σ  c_k · w_k · cos( πk(2t+1) / 2N )
        k<K
   w_0 = √(1/N),  w_k = √(2/N)
   ```

   The basis is defined for real `t`, so the same coefficients give both
   smoothing (truncate) and resampling at any framerate (evaluate off-integer).
   There is no separate interpolation step.

   A cosine basis rather than Fourier because a dance is not periodic: forcing a
   wrap between last frame and first introduces a seam whose truncation produces
   Gibbs ringing.
3. **Analyse** — differentiate the series term by term for exact joint
   velocities, and run PCA over normalised poses for the intrinsic dimension.

## Findings

Measured on a 24-second Paquita variation, 1226 frames at 50fps:

- **The variation occupies about five dimensions.** PCA over 66-dimensional pose
  space: 3 components carry 85% of the variance, 5 carry 90%, 11 carry 99%.
- **A 2:1 rhythm.** The velocity spectrum has a 1.135 Hz fundamental and a
  stronger 2.270 Hz harmonic — the signature of a left/right symmetric
  travelling step, where speed peaks once per step and a full cycle is two steps.
- **A prediction that failed.** Held balances were expected to fall out of the
  velocity minima. They do not: median shape speed is 0.925 torso-lengths per
  second throughout. A *balancé* is a travelling step, not a stillness — the
  regular oscillation is the step rhythm, which is what the 2:1 spectrum
  measures.
- **Source selection matters more than the algorithm.** Three clips through
  identical code:

  | clip | leg visibility | figure span | frames tracked |
  |---|---|---|---|
  | romantic tutu | 0.613 | 51.4% | 99.6% |
  | practice clothes | 0.863 | 34.0% | 100% |
  | classical tutu | **0.878** | **69.2%** | **100%** |

  A romantic tutu drapes over the legs; a classical tutu projects horizontally
  and never crosses them. Costume governs the skeleton, framing governs the matte.

## The turn

A pirouette is rotation about the vertical axis — exactly the axis 2D projection
discards — so it renders as a squash rather than a spin. Shoulder separation, a
rigid body dimension, collapses to 74% of its resting value in projection while
holding at 100% when measured in 3D, where the motion appears as −735° of
azimuth: a double pirouette.

Two fixes aimed at the wrong thing before that was understood (left/right label
swaps, and the low-pass cutoff). Both improved the residual; neither addressed
the cause.

`piecewise_fit.py` is the practical remedy for the 2D case — fit the turn at a
higher cutoff than the surrounding phrases and crossfade the seams, giving
1.93px residual in the turn against 1.13px elsewhere. It does not make the spin
read as a rotation, which 2D cannot do; it recovers articulation, so limbs hold
their shape through the turn instead of smearing.

`extract_world.py` and `render_3d.py` remain as diagnostic tools — they are what
established the shoulder-rigidity and azimuth measurements above — but the
published page presents the 2D work only. Metric 3D carries no global
translation, so the dancer appears to move on the spot, losing the spatial
architecture of the variation.

## Running it

```bash
python3.12 -m venv venv
./venv/bin/pip install 'mediapipe==0.10.21' 'opencv-python<5' scipy matplotlib

./venv/bin/python extract_motion.py  --video clip.mp4 --out-dir out
./venv/bin/python interpolate_motion.py --motion out/motion.json --out-dir out/fit
./venv/bin/python analyse_dynamics.py   --motion out/motion.json --out out/dynamics.png
./venv/bin/python piecewise_fit.py --motion out/motion.json --out-dir out/splice \
    --turn 18.8 20.6 --cutoff-turn 20 --cutoff-calm 6
```

MediaPipe is pinned to 0.10.x deliberately: the 1.0 release crashes on macOS ARM
inside `DrishtiMetalHelper` at graph construction, even with the CPU delegate.

## Limitations

- Monocular 2D cannot represent rotation about the vertical axis. No filter fixes
  this; only a 3D method does.
- `pose_world_landmarks` gives metric 3D but **no global translation** — the body
  configuration without its position, so the dancer appears to move on the spot.
- Loose clothing defeats the pose model's leg tracking, though not the
  segmentation mask.
- Cutoff frequency is a real parameter, not a detail. 6 Hz suits sustained
  phrases and destroys a pirouette.

## Contents

| file | purpose |
|---|---|
| `extract_motion.py` | 2D landmarks + segmentation matte |
| `extract_world.py` | metric 3D landmarks |
| `interpolate_motion.py` | cosine fit, analytic derivative, resampling |
| `piecewise_fit.py` | per-segment cutoffs with crossfaded seams |
| `analyse_dynamics.py` | velocity, held positions, PCA eigen-poses |
| `analyse_spectrum.py` | coefficient decay vs reconstruction error |
| `render_3d.py` | 3D fit and orbiting camera render |
| `build_pages.py` | wraps `site/` into a standalone `docs/` page |
| `LESSONS.md` | rules written after mistakes made here |

## Note on source material

No video, imagery or audio from any source recording is included in this
repository or on the published page. What ships is measured coordinate data and
original renderings drawn from it. The motion analysed was measured from a public
performance clip of the Paquita variation danced by Maria Khoreva.

## Licence

MIT
