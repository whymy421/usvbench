"""G20d: is the appearing cylinder visible at the trigger? (standard library only; owner decision 7A, 2026-10-04).

Ray model (the one REV-G20b occlusion_probe.py and the placement atlas occl_probe.py used for the owner's 7A / 7B
comparison, NC journal 43): ``ray_count`` true-intersection rays leave the hull COM, ray k at heading + 2 pi k /
ray_count; a ray's FIRST hit is the nearest entry point among the static occluder discs and the cylinder within
``max_range_m``; the cylinder is VISIBLE when at least one ray's first hit is the cylinder. Occluders are the key's
frozen static obstacle discs (TaskContext.static_obstacles, the same geometry the env builds, D16), copied into the
placement document; D7 intruders and native movers are not occluders here (documented limitation). Families whose
ray block uses the E4 point-return proxy (path following, station keeping, docking) see the cylinder on the ray
nearest its bearing whenever it is in range and in line of sight; this model is the stricter one (a true ray must
hit it), so ``visible`` here implies visible there.

Used twice: pool time (placement.candidate: the nominal trigger pose of each dose -> nominal_visible_at_trigger) and
run time (runtime.AppearingBridge: the realized trigger pose -> runtime.visible_at_trigger). Occluded cylinders are
KEPT (owner decision 7A: no re-placement); the stamps let the paper report the occluded share.
"""
import math

RAY_MODEL = ("true-intersection ray fan from the hull COM (ray k at heading + 2 pi k / n), first hit among the "
             "static occluder discs and the cylinder within the sensor range; visible = at least one ray's first hit "
             "is the cylinder (REV-G20b / atlas occlusion probe model)")


def ray_entry(px, py, ux, uy, cx, cy, r):
    """Distance along the unit ray (px, py) + t (ux, uy), t >= 0, to its first contact with the disc; None if it
    misses (0.0 when the origin lies inside the disc)."""
    dx, dy = cx - px, cy - py
    if dx * dx + dy * dy <= r * r:
        return 0.0
    t = dx * ux + dy * uy
    if t <= 0.0:
        return None
    perp2 = dx * dx + dy * dy - t * t
    if perp2 > r * r:
        return None
    return t - math.sqrt(max(r * r - perp2, 0.0))


def visible_rays(x, y, yaw, cylinder, occluders, max_range_m, ray_count=36):
    """Number of rays whose first hit is the cylinder (cylinder = (cx, cy, r); occluders = [(ox, oy, r), ...])."""
    cx, cy, cr = (float(v) for v in cylinder)
    count = 0
    for k in range(int(ray_count)):
        a = float(yaw) + 2.0 * math.pi * k / int(ray_count)
        ux, uy = math.cos(a), math.sin(a)
        hit = ray_entry(x, y, ux, uy, cx, cy, cr)
        if hit is None or hit > max_range_m:
            continue
        if any((e := ray_entry(x, y, ux, uy, float(ox), float(oy), float(orad))) is not None and e < hit
               for ox, oy, orad in occluders):
            continue
        count += 1
    return count


def visible(x, y, yaw, cylinder, occluders, max_range_m, ray_count=36):
    return visible_rays(x, y, yaw, cylinder, occluders, max_range_m, ray_count) > 0


__all__ = ["RAY_MODEL", "ray_entry", "visible_rays", "visible"]
