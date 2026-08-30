# Asset provenance — rescue_boat.usd

Requested by supervisor, 21 August 2026: "the USD and its source and licence".

---

## The asset

| | |
|---|---|
| File | `rescue_boat.usd` |
| Resolved from | `USVBENCH_ASSETS` environment variable |
| Vessel | ARENA RIB 8.50 m |
| Simulated mass | 300 kg (set in config, not baked into the USD) |
| Simulated as | Single rigid body |

## Source

| | |
|---|---|
| Original CAD | `ARENA RIB 8.50 METER.3dm` (Rhino) |
| Conversion | Rhino → mesh export → USD |
| Rotational inertia | Derived from the mesh by PhysX. **Not** set by `MassPropertiesCfg`, which sets mass only — see note below. |

**Note on inertia.** `MassPropertiesCfg(mass=300.0)` sets mass but leaves the
inertia tensor as computed from the mesh. Measured yaw response is consistent
with the expected order of magnitude for an 8.5 × 3.0 m hull (~2,000 kg·m²), but
the tensor has not been independently verified against the physical vessel. If
the benchmark is to be shared, this is worth pinning explicitly rather than
inheriting from geometry.

## Licence — **ACTION REQUIRED**

> **This section must be completed before the branch is shared or the work is
> published.** The CAD model was obtained during the project and its licence
> terms have not been confirmed. Do not push the USD to a public repository until
> this is resolved.

Establish and record:

1. **Origin** — where the `.3dm` came from (manufacturer, a CAD library such as
   GrabCAD, or drawn in-house).
2. **Licence terms** — whether redistribution is permitted, and under what
   conditions. GrabCAD models, for instance, are typically licensed for personal
   and educational use with redistribution restricted.
3. **Attribution** — the form the licensor requires.

If redistribution is not permitted, the options are:

- Ship a **parametric hull** generated from published dimensions instead of the
  proprietary mesh, which also removes the inertia uncertainty above.
- Ship the **conversion script** and have users supply their own CAD.
- Obtain written permission from the rights holder.

For a benchmark intended to be reproducible, the parametric option is probably
the right answer regardless: it removes a licensing dependency and makes the
vessel specification explicit rather than implicit in a mesh.

## Catamaran (P1)

The catamaran asset has the same open questions and should be recorded here too
once resolved.
