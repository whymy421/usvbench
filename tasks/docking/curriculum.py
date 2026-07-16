"""Dependency-free curriculum state for the USV docking task."""

from __future__ import annotations


class DockingCurriculum:
    """Monotonic spawn-distance curriculum driven by episodic success EMA."""

    def __init__(
        self,
        start_distance: float = 2.0,
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
        self.episodes_since_change = 0
        # Retreat guard: if the policy cannot cope with the new distance for
        # this many episodes at near-zero EMA, step back one increment.
        self.retreat_patience = 400
        self.retreat_threshold = 0.05

    def update(self, success: bool) -> float:
        """Record one finished episode and return the current spawn distance."""
        sample = 1.0 if bool(success) else 0.0
        self.success_rate_ema = (
            self.ema_decay * self.success_rate_ema
            + (1.0 - self.ema_decay) * sample
        )
        self.finished_episodes += 1
        self.episodes_since_change += 1

        if (
            self.success_rate_ema >= self.success_threshold
            and self.spawn_distance < self.max_distance
        ):
            self.spawn_distance = min(
                self.spawn_distance + self.distance_increment,
                self.max_distance,
            )
            # The new distance must re-earn the threshold from scratch; the
            # EMA reset doubles as a natural advance cooldown (~90 successful
            # episodes at decay 0.99). Without it a success streak advances
            # every update and the curriculum outruns the policy (v10: 2->25 m
            # in ~360 iterations, then success collapsed with no way back).
            self.success_rate_ema = 0.0
            self.episodes_since_change = 0
        elif (
            self.episodes_since_change >= self.retreat_patience
            and self.success_rate_ema < self.retreat_threshold
            and self.spawn_distance > self.start_distance
        ):
            self.spawn_distance = max(
                self.spawn_distance - self.distance_increment,
                self.start_distance,
            )
            self.success_rate_ema = 0.0
            self.episodes_since_change = 0
        return self.spawn_distance
