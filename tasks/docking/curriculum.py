"""Dependency-free curriculum state for the USV docking task."""

from __future__ import annotations


class DockingCurriculum:
    """Monotonic spawn-distance curriculum driven by episodic success EMA."""

    def __init__(
        self,
        start_distance: float = 3.0,
        distance_increment: float = 2.5,
        max_distance: float = 25.0,
        ema_decay: float = 0.99,
        success_threshold: float = 0.6,
    ) -> None:
        if start_distance > max_distance:
            raise ValueError("start_distance must not exceed max_distance")
        if distance_increment <= 0.0:
            raise ValueError("distance_increment must be positive")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if not 0.0 <= success_threshold <= 1.0:
            raise ValueError("success_threshold must be in [0, 1]")

        self.start_distance = float(start_distance)
        self.distance_increment = float(distance_increment)
        self.max_distance = float(max_distance)
        self.ema_decay = float(ema_decay)
        self.success_threshold = float(success_threshold)

        self.spawn_distance = self.start_distance
        self.success_rate_ema = 0.0
        self.finished_episodes = 0

    def update(self, success: bool) -> float:
        """Record one finished episode and return the current spawn distance."""
        sample = 1.0 if bool(success) else 0.0
        self.success_rate_ema = (
            self.ema_decay * self.success_rate_ema
            + (1.0 - self.ema_decay) * sample
        )
        self.finished_episodes += 1

        if (
            self.success_rate_ema >= self.success_threshold
            and self.spawn_distance < self.max_distance
        ):
            self.spawn_distance = min(
                self.spawn_distance + self.distance_increment,
                self.max_distance,
            )
        return self.spawn_distance
