"""G23e: station-keeping D7 encounter during the approach to the hold circle (sk_approach.py), CPU only.

Owner decision 2026-10-06 (CONTEXT_R6 section 3). One test class per rule:
* geometry: turn-in-place then straight reference route; meeting point at 0.45 x start-to-centre distance from
  the hold centre with the own pose there at t_c; 90 deg crossing; default radius / speed; t_c before the scoring
  window; reveal positions per tier;
* side balance: side = strata((env + episode) mod 2), exact 50/50;
* hold-circle exclusion: metadata on admitted candidates; a context whose hold circle reaches the intruder line is
  rejected with sk_intruder_path_enters_hold_circle at every nonzero dose (no re-orientation);
* not-applicable tier: BlueBoat D1 is NotApplicableByDesign, never a rejection; the pool records it separately
  (AdmissionPoolNA, digest-covered), per-dose row (0, 0); WAM-V has none;
* runtime: the default rule (d7_reveal_sync_v3 after layer B) times the SK approach like a transit route (u_ref identity, sk_approach stamp block);
  d7_exposure accepts the stamp and rejects tampered ones; the retired hold design is refused;
* identity guards: build_transit_plan(align_first=False) and generate_encounter defaults are unchanged.
"""
import importlib.util
import json
import math
import sys
import unittest
from dataclasses import replace

import fixtures as f

d = f.d
SK = d.adapters.sk_approach
BB, WAMV = f.VESSELS[0], f.VESSELS[1]
DT = 1.0 / 120.0
V2 = d.release.DEFAULT_RULE      # G23e layer B: the default rule is d7_reveal_sync_v3 (name kept for the diff)
SCRIPTS = f.PACKAGE.parents[3] / "scripts"


def _d7_exposure():
    if "g23e_d7_exposure" not in sys.modules:
        spec = importlib.util.spec_from_file_location("g23e_d7_exposure", SCRIPTS / "d7_exposure.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["g23e_d7_exposure"] = module
        spec.loader.exec_module(module)
    return sys.modules["g23e_d7_exposure"]


def context(vessel, **overrides):
    return f.context("station_keeping", f.envelope(vessel), **overrides)


def ladder(vessel, env_row=0, episode=0, **overrides):
    ctx = context(vessel, **overrides)
    key = f.key("station_keeping", env=env_row, episode=episode)
    return ctx, key, d.build_ladder(ctx, key, f.suite(ctx.envelope))


def plan_of(ctx):
    return d.adapters._nominal_plan(d.PROFILES["station_keeping"], ctx, f.suite(ctx.envelope))


def run(assignment, pose_fn, dt=DT, until=None, rule=V2):
    cfg = d.DynamicRuntimeConfig(motion_rule=rule)
    adapter = d.make_adapter("station_keeping")
    row = assignment.candidate.request.encounter.key.env
    adapter.reset((row,), cfg, {row: assignment})
    adapter.activate((row,), cfg)
    end = assignment.candidate.request.encounter.duration_s if until is None else until
    samples, i = [], 0
    while i * dt <= end + 1e-12:
        t = i * dt
        out = adapter.update(d.RuntimeFrame(((row, t),), ((row, pose_fn(t)),)), cfg)[row]
        samples.append((t, tuple((s.x, s.y, s.physical, s.visible) for s in out)))
        i += 1
    return adapter.finish(row, "timeout"), samples


class TestGeometry(unittest.TestCase):
    def test_reference_route_turns_in_place_then_runs_straight(self):
        for vessel in (BB, WAMV):
            ctx = context(vessel, initial_pose=d.Pose(-12 * f.scale(f.envelope(vessel)), 0.0, 2.5, 0.0))
            plan = plan_of(ctx)
            moving = [s for s in plan.states if s.pose.speed > 0.0]
            self.assertTrue(moving)
            first = plan.states.index(moving[0])
            for s in plan.states[:first]:                 # turn in place: no translation
                self.assertEqual((s.pose.x, s.pose.y), (ctx.initial_pose.x, ctx.initial_pose.y))
            for s in plan.states[first:]:                 # then the straight start -> centre line (y = 0)
                self.assertLess(abs(s.pose.y), 1e-9)
                self.assertEqual(s.yaw_rate_rps, 0.0)

    def test_meeting_point_and_crossing(self):
        for vessel in (BB, WAMV):
            for yaw in (0.0, 1.3, -2.9):
                env = f.envelope(vessel)
                ctx, key, lad = ladder(vessel, initial_pose=d.Pose(-12 * f.scale(env), 0.0, yaw, 0.0))
                plan = plan_of(ctx)
                hold = ctx.hold_circle
                d_start = math.hypot(hold.x - ctx.initial_pose.x, hold.y - ctx.initial_pose.y)
                for dose, cand in enumerate(lad):
                    if isinstance(cand, SK.NotApplicableByDesign):
                        continue
                    enc = cand.request.encounter
                    t_c = enc.meta("conflict_time_s")
                    own = plan.pose_at(t_c)
                    self.assertAlmostEqual(math.hypot(own.x - hold.x, own.y - hold.y), 0.45 * d_start, delta=1e-9)
                    self.assertLess(math.hypot(own.x - enc.meta("sk_meeting_x_m"), own.y - enc.meta("sk_meeting_y_m")),
                                    SK.SK_MEETING_TOL_M)
                    self.assertAlmostEqual(enc.meta("sk_meeting_x_m"), ctx.initial_pose.x + 0.55 * d_start, delta=1e-12)
                    self.assertEqual(enc.meta("sk_design"), SK.SK_DESIGN)
                    self.assertLess(t_c, ctx.settled_phase_start_s)         # never during the hold
                    if dose == 0:
                        continue
                    obj = enc.intruders[0]
                    self.assertEqual(obj.path_type, "crossing")
                    # 90 deg: intruder velocity perpendicular to the own heading at t_c, default speed / radius
                    self.assertLess(abs(obj.vx * math.cos(own.yaw) + obj.vy * math.sin(own.yaw)), 1e-12)
                    self.assertAlmostEqual(math.hypot(obj.vx, obj.vy), 0.5, delta=1e-12)
                    self.assertAlmostEqual(obj.radius_m, 0.25 * env.beam_m, delta=1e-12)
                    self.assertAlmostEqual(abs(enc.meta("relative_heading_rad")), math.pi / 2, delta=1e-12)
                    # nominal reveal on the straight cruise-speed approach, TTC = dose x t_avoid
                    self.assertAlmostEqual(t_c - obj.reveal_s, d.DOSE_TTC_RATIOS[dose] * env.t_avoid_s, delta=1e-9)
                    self.assertTrue(plan.is_constant_velocity(obj.reveal_s, t_c))
                    self.assertGreaterEqual(obj.reveal_s, 0.0)

    def test_reveal_distance_per_tier_matches_the_design_arithmetic(self):
        for vessel in (BB, WAMV):
            env = f.envelope(vessel)
            ctx, key, lad = ladder(vessel)
            plan = plan_of(ctx)
            d_start = 12 * f.scale(env)
            for dose, cand in enumerate(lad):
                if dose == 0 or isinstance(cand, SK.NotApplicableByDesign):
                    continue
                obj = cand.request.encounter.intruders[0]
                travelled = plan.distance_at(obj.reveal_s)
                expected = 0.55 * d_start - d.DOSE_TTC_RATIOS[dose] * env.t_avoid_s * env.reference_speed_mps
                self.assertAlmostEqual(travelled, expected, delta=1e-9)


class TestSideBalance(unittest.TestCase):
    def test_side_by_scene_key_is_exactly_balanced(self):
        counts = {1: 0, -1: 0}
        for env_row in range(8):
            for episode in range(3):
                ctx, key, lad = ladder(WAMV, env_row=env_row, episode=episode)
                sides = {c.request.encounter.meta("side") for c in lad if not isinstance(c, Exception)}
                self.assertEqual(len(sides), 1)                       # the same side at every dose
                side = sides.pop()
                self.assertEqual(side, (1, -1)[(env_row + episode) % 2])
                counts[side] += 1
        self.assertEqual(counts, {1: 12, -1: 12})


class TestHoldExclusion(unittest.TestCase):
    def test_admitted_candidates_keep_the_margin(self):
        for vessel in (BB, WAMV):
            env = f.envelope(vessel)
            ctx, key, lad = ladder(vessel)
            for cand in lad[1:]:
                if isinstance(cand, SK.NotApplicableByDesign):
                    continue
                enc = cand.request.encounter
                line, required = SK.hold_geometry(ctx.hold_circle, env, enc.intruders[0])
                self.assertEqual(required, ctx.hold_circle.radius + enc.intruders[0].radius_m + 0.5 * env.beam_m)
                self.assertEqual(enc.meta("sk_hold_line_distance_m"), line)
                self.assertGreater(enc.meta("sk_hold_clearance_m"), 0.0)

    def test_path_through_the_hold_circle_is_rejected_without_reorientation(self):
        for vessel in (BB, WAMV):
            env = f.envelope(vessel)
            u = f.scale(env)
            big = d.Circle(0.0, 0.0, 5.3 * u)            # reaches the intruder line (~0.45 x 12 u - offset)
            ctx, key, lad = ladder(vessel, hold_circle=big)
            reasons = [getattr(c, "reason", None) for c in lad[1:]]
            first = 2 if vessel == BB else 1
            self.assertEqual(reasons[first - 1:], [SK.SK_HOLD_REJECTION] * (5 - first))
            pool = d.admit_ladders([(key, lad)])
            self.assertEqual(pool.keys_admitted, 0)
            self.assertIn(SK.SK_HOLD_REJECTION, {r.reason for r in pool.rejections})

    def test_validate_refuses_a_path_into_the_margin(self):
        ctx, key, lad = ladder(WAMV)
        cand = lad[2]
        cand.validate()
        enc = cand.request.encounter
        obj = enc.intruders[0]
        moved = replace(obj, x0=ctx.hold_circle.x - obj.vx * 10.0, y0=ctx.hold_circle.y - obj.vy * 10.0)
        forged = replace(cand, request=replace(cand.request, encounter=replace(enc, intruders=(moved,))))
        with self.assertRaises(ValueError):
            SK.validate_candidate(forged)


class TestNotApplicableTier(unittest.TestCase):
    def test_blueboat_d1_is_not_applicable_wamv_has_none(self):
        _, key, lad = ladder(BB)
        self.assertIsInstance(lad[1], SK.NotApplicableByDesign)
        self.assertEqual(lad[1].reason, SK.SK_NOT_APPLICABLE)
        self.assertTrue(all(not isinstance(c, Exception) for c in (lad[0],) + lad[2:]))
        self.assertEqual({c.request.encounter.meta("first_applicable_dose") for c in (lad[0],) + lad[2:]}, {2})
        _, _, lad_w = ladder(WAMV)
        self.assertTrue(all(not isinstance(c, Exception) for c in lad_w))
        self.assertEqual({c.request.encounter.meta("first_applicable_dose") for c in lad_w}, {1})
        # BlueBoat D1: the straight cruise-speed run before M (0.55 x 12 u less the acceleration distance) is
        # shorter than 4 t_avoid u_ref, so the D1 reveal would precede the straight approach
        ctx = context(BB)
        env = ctx.envelope
        plan = plan_of(ctx)
        cruise = next(s for s in plan.states if s.pose.speed >= env.reference_speed_mps)
        straight = 0.55 * 12 * f.scale(env) - plan.distance_at(cruise.time_s)
        self.assertLess(straight, 4 * env.t_avoid_s * env.reference_speed_mps)
        self.assertGreaterEqual(straight, 3 * env.t_avoid_s * env.reference_speed_mps)

    def test_pool_records_not_applicable_separately_from_rejections(self):
        ctx, key, lad = ladder(BB)
        pool = d.admit_ladders([(key, lad)])
        self.assertIsInstance(pool, d.adapters.AdmissionPoolNA)
        self.assertEqual(pool.keys_admitted, 1)
        self.assertEqual(pool.rejections, ())
        self.assertEqual(pool.per_dose_counts, ((1, 1), (0, 0), (1, 1), (1, 1), (1, 1)))
        self.assertEqual([(r.key, r.dose, r.reason) for r in pool.not_applicable], [(key, 1, SK.SK_NOT_APPLICABLE)])
        with self.assertRaises(KeyError):
            pool.assignment(key, 1)
        grid = d.admit_grid([(ctx, key, f.suite(ctx.envelope))])
        self.assertEqual(grid.pool_digest, pool.pool_digest)
        self.assertEqual(grid.not_applicable, pool.not_applicable)
        # a pool without such tiers stays an AdmissionPool (unchanged manifests of every other task)
        ctx_w, key_w, lad_w = ladder(WAMV)
        self.assertIs(type(d.admit_ladders([(key_w, lad_w)])), d.AdmissionPool)

    def test_not_applicable_never_breaks_the_joint_ladder(self):
        ctx, key, lad = ladder(BB)
        broken = lad[:3] + (d.PlacementError("forced_rejection"),) + lad[4:]
        pool = d.admit_ladders([(key, broken)])
        self.assertEqual(pool.keys_admitted, 0)
        self.assertNotIn(SK.SK_NOT_APPLICABLE, {r.reason for r in pool.rejections})
        self.assertEqual({r.dose for r in pool.rejections if r.reason == "joint_ladder_member_rejected"}, {0, 2, 4})
        self.assertEqual(len(pool.not_applicable), 1)

    def test_generator_refuses_a_dose_below_the_first_applicable(self):
        ctx, key, lad = ladder(BB)
        plan = plan_of(ctx)
        t_c = lad[0].request.encounter.meta("conflict_time_s")
        with self.assertRaises(ValueError):
            d.generate_encounter(key, "station_keeping", 1, ctx.envelope, plan, t_c, ctx.deadline_s, first_dose=2)


class TestRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {}
        for vessel in (BB, WAMV):
            ctx, key, lad = ladder(vessel)
            cls.cases[vessel] = (ctx, key, d.admit_ladders([(key, lad)]))

    def test_nominal_own_ship_reveals_at_the_dose_and_stamps_the_sk_block(self):
        e = _d7_exposure()
        for vessel, (ctx, key, pool) in self.cases.items():
            plan = plan_of(ctx)
            for dose in ((2, 4) if vessel == BB else (1, 4)):
                a = pool.assignment(key, dose)
                enc = a.candidate.request.encounter
                obj = enc.intruders[0]
                t_c = enc.meta("conflict_time_s")
                stamp, samples = run(a, lambda t, p=plan: p.pose_at(min(t, p.end_s)), until=t_c + 1.0)
                rel = stamp["payload"]["runtime"]["release"]
                self.assertTrue(rel["applies"])
                self.assertEqual(rel["plan_kind"], SK.SK_PLAN_KIND)
                self.assertEqual(rel["cause"], "ttc_reveal")
                self.assertTrue(rel["snapped_to_pinned"])
                self.assertEqual(rel["release_time_s"], obj.reveal_s)            # u_ref identity
                self.assertTrue(rel["dose_achieved"])
                block = rel["sk_approach"]
                self.assertEqual(block["design"], SK.SK_DESIGN)
                self.assertEqual(block["first_applicable_dose"], 2 if vessel == BB else 1)
                self.assertEqual(block["side"], enc.meta("side"))
                # hidden before the reveal, the pinned motion from the reveal on
                for t, s in samples:
                    if t < rel["release_tick_time_s"]:
                        self.assertFalse(s[0][2])
                    else:
                        self.assertTrue(s[0][2])
                        self.assertEqual(s[0][:2], obj.position(t))
                reading = e.runtime_exposure(json.loads(json.dumps(stamp)))
                self.assertTrue(reading["released"])

    def test_tampered_or_retired_sk_stamps_are_rejected(self):
        e = _d7_exposure()
        ctx, key, pool = self.cases[WAMV]
        plan = plan_of(ctx)
        a = pool.assignment(key, 3)
        stamp, _ = run(a, lambda t, p=plan: p.pose_at(min(t, p.end_s)), dt=0.05,
                       until=a.candidate.request.encounter.meta("conflict_time_s") + 1.0)
        good = json.loads(json.dumps(stamp))
        e.runtime_exposure(good)
        for mutate in (lambda r: r.__setitem__("applies", False),
                       lambda r: r.__setitem__("plan_kind", "hold"),
                       lambda r: r.pop("sk_approach"),
                       lambda r: r["sk_approach"].__setitem__("side", 0),
                       lambda r: r["sk_approach"].__setitem__("crossing_angle_rad", 0.0),
                       lambda r: r["sk_approach"].__setitem__("meeting_fraction", 0.5),
                       lambda r: r["sk_approach"].__setitem__("hold_clearance_m", -0.1),
                       lambda r: r["sk_approach"].__setitem__("first_applicable_dose", 4)):
            bad = json.loads(json.dumps(stamp))
            mutate(bad["payload"]["runtime"]["release"])
            with self.assertRaises(e.ExposureError):
                e.runtime_exposure(bad)

    def test_constants_agree_with_the_validator(self):
        e = _d7_exposure()
        self.assertEqual(e.SK_DESIGN, SK.SK_DESIGN)
        self.assertEqual(e.SK_PLAN_KIND, SK.SK_PLAN_KIND)
        self.assertEqual(e.SK_MEETING_FRACTION, SK.SK_MEETING_FRACTION)
        self.assertEqual(e.SK_CROSSING_ANGLE_RAD, SK.SK_CROSSING_ANGLE_RAD)
        self.assertEqual(e.NOT_APPLICABLE, SK.SK_NOT_APPLICABLE)

    def test_header_not_applicable_doses(self):
        e = _d7_exposure()
        header = {"per_dose_counts": [[4, 4, 0], [0, 0, 0], [4, 4, 0], [4, 4, 0], [4, 4, 0]],
                  "not_applicable_doses": [1]}
        self.assertEqual(e.not_applicable_doses(header), (1,))
        self.assertEqual(e.not_applicable_doses({"per_dose_counts": header["per_dose_counts"]}), ())
        for bad in ([0], [2, 1], [1, 1], ["1"]):
            with self.assertRaises(e.ExposureError):
                e.not_applicable_doses(dict(header, not_applicable_doses=bad))
        with self.assertRaises(e.ExposureError):
            e.not_applicable_doses(dict(header, per_dose_counts=[[4, 4, 0]] * 5))


class TestIdentityGuards(unittest.TestCase):
    def test_align_first_default_keeps_the_plan(self):
        for vessel in (BB, WAMV):
            env = f.envelope(vessel)
            for yaw in (0.0, 0.7, -2.0):
                pose = d.Pose(0.0, 0.0, yaw, 0.0)
                targets = (d.Circle(20.0, 5.0, 1.0),)
                a = d.build_transit_plan(pose, env, targets, 0.25, 200.0, stop_at_end=True)
                b = d.build_transit_plan(pose, env, targets, 0.25, 200.0, stop_at_end=True, align_first=False)
                self.assertEqual(a, b)
                c = d.build_transit_plan(pose, env, targets, 0.25, 200.0, stop_at_end=True, align_first=True)
                if yaw != math.atan2(5.0, 20.0):
                    self.assertNotEqual(a.states, c.states)

    def test_settled_clock_rule_is_retired(self):
        self.assertNotIn("settled_clock", d.adapters.SITE_RULES)
        self.assertEqual(d.PROFILES["station_keeping"].site_rule, SK.SK_SITE_RULE)
        self.assertEqual(d.PROFILES["station_keeping"].kinds, ("crossing",))
        ctx, key, lad = ladder(WAMV)
        cand = lad[2]
        enc = cand.request.encounter
        old = replace(enc, metadata=tuple(p for p in enc.metadata if p[0] != "sk_design"))
        forged = replace(cand, request=replace(cand.request, encounter=old))
        with self.assertRaises(ValueError):
            forged.validate()


if __name__ == "__main__":
    unittest.main(verbosity=2)
