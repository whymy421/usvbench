"""Plain-assert tests for the dependency-free docking curriculum."""

from curriculum import DockingCurriculum


def test_v3_defaults() -> None:
    curriculum = DockingCurriculum()
    assert curriculum.start_distance == 2.0
    assert curriculum.spawn_distance == 2.0
    assert curriculum.distance_increment == 2.5
    assert curriculum.success_threshold == 0.6
    assert curriculum.max_distance == 25.0


def test_no_advance_below_threshold() -> None:
    curriculum = DockingCurriculum(ema_decay=0.5, success_threshold=0.75)
    assert curriculum.update(True) == 2.0
    assert curriculum.success_rate_ema == 0.5


def test_advance_at_threshold() -> None:
    curriculum = DockingCurriculum(ema_decay=0.5, success_threshold=0.5)
    assert curriculum.update(True) == 4.5
    # The new distance must re-earn the threshold from scratch.
    assert curriculum.success_rate_ema == 0.0


def test_no_advance_streak_runaway() -> None:
    # v10 regression: at production decay (0.99) a success streak must NOT
    # advance on every update -- the post-advance EMA reset forces ~91
    # consecutive successes before the next advance.
    curriculum = DockingCurriculum(ema_decay=0.99, success_threshold=0.6)
    for _ in range(120):
        curriculum.update(True)  # reaches 0.6 once (~91 updates), advances
    assert curriculum.spawn_distance == 4.5  # exactly one advance in 120


def test_retreat_after_patience() -> None:
    curriculum = DockingCurriculum(ema_decay=0.5, success_threshold=0.5)
    curriculum.update(True)  # advance to 4.5
    for _ in range(curriculum.retreat_patience):
        curriculum.update(False)
    assert curriculum.spawn_distance == 2.0  # backed off, floored at start
    assert curriculum.success_rate_ema == 0.0


def test_cap_at_25_metres() -> None:
    curriculum = DockingCurriculum(ema_decay=0.0, success_threshold=0.6)
    for _ in range(20):
        curriculum.update(True)
    assert curriculum.spawn_distance == 25.0


def test_distance_monotonic_within_patience() -> None:
    # Short mixed sequences (fewer than retreat_patience episodes) never
    # retreat, so distance stays monotonic over them.
    curriculum = DockingCurriculum(ema_decay=0.5, success_threshold=0.6)
    distances = [curriculum.spawn_distance]
    for success in (False, True, True, False, False, True, True, True, False):
        distances.append(curriculum.update(success))
    assert all(after >= before for before, after in zip(distances, distances[1:]))


def main() -> None:
    test_v3_defaults()
    test_no_advance_below_threshold()
    test_advance_at_threshold()
    test_no_advance_streak_runaway()
    test_retreat_after_patience()
    test_cap_at_25_metres()
    test_distance_monotonic_within_patience()
    print("curriculum tests: PASS")


if __name__ == "__main__":
    main()
