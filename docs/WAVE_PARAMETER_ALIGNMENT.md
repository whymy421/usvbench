# Wave Implementation and Parameter Alignment

Both sides use the latest shared wave implementation in
`tasks/_shared/waves.py`. The older task-local JONSWAP implementations are not
the basis for later experiments, comparisons, or reported results.

## JONSWAP Definitions

### Significant Wave Height `Hs`

Irregular-wave height is represented by `Hs` in metres:

```text
Hs = 4 sqrt(m0)
m0 = integral S(f) df
```

Here `S(f)` is the wave spectrum and `m0` is its zeroth moment. The shared
implementation computes `m0` over the actual discrete frequency band and
renormalizes the spectrum so that `4 sqrt(m0) = Hs`. Therefore `hs_min_m` and
`hs_max_m` are the final realized significant wave-height bounds, not nominal
uncorrected scale parameters.

### Peak Period `Tp`

The spectral peak period is represented by `Tp` in seconds:

```text
Tp = 1 / fp
```

`fp` is the target JONSWAP peak frequency in Hz. `Tp` is not the mean,
zero-crossing, or energy period. Because the spectrum uses finite discrete
frequencies, the largest sampled spectral value can be slightly offset from
`fp`. Record that sampled bin as `f_peak_bin` or `T_peak_bin = 1 / f_peak_bin`
instead of conflating it with the configured `Tp`.

### Current Defaults and Sampling

The current HazardNav shared implementation uses the following code defaults.
They are implementation defaults, not frozen training or certification levels.

| Parameter | Default | Meaning |
|---|---:|---|
| `gamma` | `3.3` | Fixed certification peak-enhancement factor; exploratory runs may override it explicitly. |
| `f_min_hz` | `0.10` | Lower frequency-band bound. |
| `f_max_hz` | `1.60` | Upper frequency-band bound. |
| `spread_deg` | `30` | Total directional spread, equal to `15` degrees on either side of the mean direction. |
| `n_components` | `30` | Number of frequency components. |
| `buoyancy_stations` | `6` | Number of distributed hull buoyancy samples. |

Certification fixes `gamma=3.3`. Training and evaluation should sample `Hs` and
`Tp` with `sampling_mode="levels"` and explicit `hs_levels_m` and
`tp_levels_s`; certification may set `gamma_levels` to `(3.3,)`. Exploratory
uniform sampling must be recorded explicitly. The repository does not freeze
unconfirmed `Hs x Tp` packages as a baseline.

When `direction_deg=None`, the mean propagation direction is deterministically
pseudo-randomized in `[0, 360 degrees)`. An explicit value fixes that direction
for every environment.

### Reproducible Phase and Direction Seeds

Phases, mean propagation direction, per-component directional spread, and sea
state parameters are determined by the pure function:

```text
random_value = f(eval_seed, environment_index, episode_index, stream_index)
```

The reference certification seed is `eval_seed=42`, matching the default
`--seed` in `scripts/eval_hazard_nav.py`. Reports must record the actual seed.
`environment_index` starts at zero. Each environment maintains its own
`episode_index`, starting at zero on the first reset and increasing by one on
each subsequent reset.

Each environment uses an independent CPU `torch.Generator` seeded from that
integer tuple; the global Torch RNG is neither read nor advanced. Consequently,
the same evaluation seed, environment index, and episode index produce the same
`Hs/Tp/gamma`, phases, and directions. Each episode's wave time starts at local
`t=0` on reset and is independent of the previous episode's actual end time.
Changing the order of reset subsets does not change the result. Obstacle layouts
use the same `(eval_seed, environment_index, episode_index)` protocol so paired
policies see the same episode stream.

### Zero-Amplitude Identity

When `mode="calm"`, Airy `height_m=0`, or JONSWAP `hs_max_m=0`, the factory
returns `CalmWater` directly. This path samples no buoyancy stations and applies
no orbital or surface velocity, so it is exactly the calm-water path rather than
an approximation obtained by computing a zero-amplitude wave.

## Wave-Field Discretization

The default JONSWAP field is the sum of **30 frequency components**:
`n_components = 30`. Each component gets its amplitude from the JONSWAP
spectrum and has an independent random phase and propagation direction.

```text
eta(x, y, t) = sum_n a_n cos(k_n (dx_n x + dy_n y) - omega_n t + phi_n)
a_n = sqrt(2 S(f_n) Delta_f_n)
```

Wavenumbers use the deep-water relation `k_n = omega_n^2 / g`. Frequencies are
not strictly uniform: a fixed small jitter is added to the midpoint grid with
nominal `df=(f_max-f_min)/30`, and each `Delta_f_n` is derived from adjacent
midpoints. The spectrum normalization uses those same per-bin widths, so
`4 sqrt(m0) = Hs` remains exact.

### Directional Spread

`spread_deg` is the total width, not a one-sided angle. Each component samples
uniformly from `[-spread_deg/2, +spread_deg/2]` and adds that offset to the
mean direction. This is not a cosine-power (`cos^s`) directional distribution;
`spread_deg=30 degrees` explicitly means `+/-15 degrees`.

### Validity Domain and Deep-Water Assumption

HazardNav requires every possible configured combination to satisfy:

```text
lambda_p = g Tp^2 / (2 pi)
Hs / lambda_p <= 0.05
```

This is the steepness limit of the linear deep-water model; invalid settings
raise during field construction. No finite-depth dispersion correction is
included. If water depth `h` is not clearly greater than `lambda_p/2`, confirm
applicability separately and do not call the result a shallow-water certification.

The 30 points above are **frequency components**, not hull buoyancy samples. The
default frequency jitter is `0.22` times nominal `df`; it is a fixed frequency
design parameter, not a per-episode random draw.

## Hull-Wave Coupling

The current hull model uses **6 distributed buoyancy samples**:
`buoyancy_stations = 6`, not 8.

Each station computes submergence and buoyancy from the local surface directly
below it. Summed station forces naturally produce heave, roll, and pitch. The
horizontal wave contribution enters drag through relative hull-to-water-particle
velocity, and vertical damping uses the difference between hull vertical speed
and surface vertical speed.

No authored wave-force or wave-moment gain is added. Wave response comes from the
existing buoyancy, hydrostatic restoring, and damping parameters.

| Quantity | Current default | Purpose |
|---|---:|---|
| JONSWAP frequency components | 30 | Discretize and synthesize the irregular field. |
| Hull buoyancy samples | 6 | Compute local submergence, buoyancy, roll, and pitch moments. |

## Regular-Wave Note

For an Airy regular wave, crest-to-trough height is `H` and period is `T`.
Regular-wave `H` is not the same definition as irregular-sea `Hs` and must not
share a result field. An energy-equivalent regular-wave comparison uses:

```text
H = Hs / sqrt(2)
T = Tp
```

If `H = Hs` is used only to match the numeric height, label it as such; it is not
an energy-equivalent comparison.

## Experiment Records

Every JONSWAP experiment should record at least:

```text
model=JONSWAP
eval_seed=<integer>
environment_index=<integer>
episode_index=<integer>
Hs=<value or range> m
Tp=<value or range> s
gamma=<value or range>
sampling_mode=<uniform or levels>
Hs_levels=<explicit list when levels>
Tp_levels=<explicit list when levels>
f_band=[f_min_hz, f_max_hz] Hz
max_steepness=0.05
n_components=30
spread=<value> deg
direction=<value or random>
buoyancy_stations=6
```

With these fields recorded, matching configurations produce the same wave
definition, field discretization, and hull-coupling convention.
