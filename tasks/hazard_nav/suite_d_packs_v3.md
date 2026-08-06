# Suite D: remaining dynamics axes protocol v3

Frozen 2026-08-06 BEFORE any v3 evaluation ran, under the same rule as the
imbalance v1/v2 protocols: pack boundaries and expectations may not move after
seeing results.

Suite D uses the forced-crossing carrier task and changes one plant axis at a
time. Each training id samples its train pack uniformly per environment at
every episode reset. Evaluation must disable that choice tuple and apply one
scalar value at a time. For every value and evaluation seed, report success
rate, collision episodes/rate, and path-length statistics separately; do not
pool values before reporting them.

## Mass scale

- Train: `{0.80, 1.00, 1.30}`
- Interpolation: `{0.90, 1.10, 1.20}`
- Extrapolation: `{1.75, 3.00, 5.00}`

`mass_scale` multiplies both effective mass and inertia. The BlueBoat asset
authors its rigid-body mass, so the environment leaves the USD untouched and
divides every assembled external force and torque by this scale. This is exact
for free-body force/torque acceleration and keeps gravity mass-independent; it
does not pretend to change PhysX contact impulse inertia.

The 0.80 lower train boundary represents a lightly equipped hull; 1.30 is a
30% increase, equivalent to about 10.4 kg against the registry's 34.6 kg
displaced-water scale, and covers payload or modest water ingress. The 1.75
extrapolation boundary is an overloaded/flooded craft. 3.00 and 5.00 are severe
flooding or tow-load fault envelopes, 2.31x and 3.85x the training maximum,
chosen to force a visible transient-response change rather than repeat v1's
timid imbalance extrapolation.

Falsifiable expectation (pre-registered): success and median path remain close
to nominal through interpolation, then timeouts rise monotonically over
`{1.75, 3.00, 5.00}` because acceleration and turning response fall; if SR is
still above 90% at 5.00, mass is declared non-discriminating on this carrier.

## Drag scale

- Train: `{0.75, 1.00, 1.50}`
- Interpolation: `{0.875, 1.20, 1.35}`
- Extrapolation: `{3.00, 8.00, 20.00}`

`drag_scale` multiplies the plant's planar surge/sway and yaw linear and
quadratic drag loads. It does not scale buoyancy, hydrostatic restoring terms,
heave damping, or roll/pitch rate damping. The 0.75 lower boundary represents a
clean, lightly wetted hull; 1.50 covers roughness, early biofouling, or weeds.
At 3.00 the craft is heavily fouled. 8.00 represents a weed/debris-entangled
hull, and 20.00 is an extreme debris, ice-slush, or dragging-line fault. Under
quadratic-drag dominance, those extrapolation boundaries reduce equilibrium
speed to about 0.58, 0.35, and 0.22 of the nominal value.

Falsifiable expectation (pre-registered): peak speed falls monotonically with
drag; SR is stable in interpolation but drops below 50% by 20.00 through slow
transits and reduced yaw authority. If SR exceeds 90% at 20.00, drag is
declared non-discriminating on this carrier.

## Thrust-cap scale

- Train: `{0.75, 0.875, 1.00}`
- Interpolation: `{0.8125, 0.90, 0.95}`
- Extrapolation: `{0.50, 0.20, 0.05}`

`thrust_cap_scale` multiplies forward and reverse thrust maxima only; yaw
torque remains the independently configured yaw channel. 1.00 is the rated
cap. The 0.75 train boundary covers sustained voltage sag, propeller damage,
or ordinary motor wear. 0.50 is a severe battery/propulsion derating, 0.20 is
limp-home power, and 0.05 is a near-loss-of-propulsion fault: one twentieth of
rated surge and 15x below the training reduction from nominal. These values
separate loss of surge authority from the imbalance axis's forced veer.

Falsifiable expectation (pre-registered): peak speed and SR fall monotonically
as the cap falls; interpolation stays near nominal, 0.20 produces a majority
of timeouts, and SR is below 10% at 0.05. If SR remains above 90% at 0.05, the
carrier or evaluator is insensitive to surge authority and this axis fails.

## Motor time constant

- Train: `{0.00, 0.10, 0.25}`
- Interpolation: `{0.05, 0.15, 0.20}`
- Extrapolation: `{1.00, 3.00, 10.00}`

`motor_tau_s` is the first-order time constant applied independently to the
commanded surge thrust and yaw torque at each 120 Hz physics substep (policy
commands arrive at 60 Hz). Zero is the exact instantaneous legacy path.
0.10--0.25 s spans ESC, motor, propeller and
short drivetrain response. 1.00 s represents a badly rate-limited controller
or power system, 3.00 s a severe propulsion-control fault, and 10.00 s a nearly
unresponsive actuator: 40x the training maximum and long enough to span a
substantial fraction of a gate approach. The per-environment applied state is
cleared on every episode reset so no motor memory crosses episode boundaries.

Falsifiable expectation (pre-registered): the measured 63% step-response time
increases monotonically with `motor_tau_s`; interpolation retains high SR,
while collision and timeout rates rise over `{1.00, 3.00, 10.00}` and SR is
below 50% at 10.00. If SR stays above 90% at 10.00, actuator lag is declared
non-discriminating on this carrier.

## Registered training carriers

- `Isaac-USV-HazardCrossMass-Direct-v1`
- `Isaac-USV-HazardCrossDrag-Direct-v1`
- `Isaac-USV-HazardCrossThrust-Direct-v1`
- `Isaac-USV-HazardCrossTau-Direct-v1`

The discount-coupling control uses the same forced-crossing carrier and no
dynamics choice pack: `Isaac-USV-HazardPbrsGamma-Direct-v1` sets only
`pbrs_correct=True`, while `Isaac-USV-HazardPbrsNoGamma-Direct-v1` sets only
`pbrs_correct=False`.
