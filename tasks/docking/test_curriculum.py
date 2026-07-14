"""Plain-assert tests for the dependency-free docking curriculum."""

from curriculum import DockingCurriculum


def test_no_advance_below_threshold() -> None:
    curriculum = DockingCurriculum(ema_decay=0.5, success_threshold=0.75)
    assert curriculum.update(True) == 3.0
    assert curriculum.success_rate_ema == 0.5


def test_advance_at_threshold() -> None:
    curriculum = DockingCurriculum(ema_decay=0.5, success_threshold=0.5)
    assert curriculum.update(True) == 5.5
    assert curriculum.success_rate_ema == 0.5


def test_cap_at_25_metres() -> None:
    curriculum = DockingCurriculum(ema_decay=0.0, success_threshold=0.6)
    for _ in range(20):
        curriculum.update(True)
    assert curriculum.spawn_distance == 25.0


def test_distance_is_monotonic() -> None:
    curriculum = DockingCurriculum(ema_decay=0.5, success_threshold=0.6)
    distances = [curriculum.spawn_distance]
    for success in (False, True, True, False, False, True, True, True, False):
        distances.append(curriculum.update(success))
    assert all(after >= before for before, after in zip(distances, distances[1:]))


def main() -> None:
    test_no_advance_below_threshold()
    test_advance_at_threshold()
    test_cap_at_25_metres()
    test_distance_is_monotonic()
    print("curriculum tests: PASS")


if __name__ == "__main__":
    main()
