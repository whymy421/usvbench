"""Pre-construction slot ownership. No writes to task tensors or simulator APIs."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CapacityPlan:
    base_max: int
    dynamic_reserve: int
    appearing_reserve: int = 0
    capacity: int = 0

    def __post_init__(self):
        values = (self.base_max, self.dynamic_reserve, self.appearing_reserve, self.capacity)
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError("slot counts must be nonnegative integers")
        if self.capacity < self.required:
            raise ValueError("capacity must cover base maximum plus BOTH condition reserves")

    @property
    def required(self):
        return self.base_max + self.dynamic_reserve + self.appearing_reserve

    @property
    def dynamic_slots(self):
        return tuple(range(self.base_max, self.base_max + self.dynamic_reserve))

    @property
    def appearing_slots(self):
        return tuple(range(self.base_max + self.dynamic_reserve, self.required))

    def validate_counts(self, base_count, dynamic_count, appearing_count=0):
        counts = (base_count, dynamic_count, appearing_count)
        limits = (self.base_max, self.dynamic_reserve, self.appearing_reserve)
        if any(type(n) is not int or n < 0 or n > limit for n, limit in zip(counts, limits)):
            raise ValueError("episode slot count exceeds its reserved ownership")


class SlotState:
    """Small CPU reference bridge; update only owned moving slots.

    Static insertions write BOTH current and persistent base centers, preventing
    C1 finding 43's (8,9)->(0,0) regression even when native motion runs later.
    """
    def __init__(self, plan):
        self.plan = plan
        self.base = [None] * plan.capacity
        self.current = [None] * plan.capacity
        self.motions = {}

    def insert_static(self, slot, circle):
        self._check(slot)
        self.base[slot] = circle
        self.current[slot] = circle
        self.motions.pop(slot, None)

    def insert_motion(self, slot, motion):
        from .model import Circle
        self._check(slot)
        self.motions[slot] = motion
        x, y = motion.position(0.0)
        self.base[slot] = Circle(x, y, motion.radius_m)
        self.current[slot] = self.base[slot]

    def update(self, time_s):
        from .model import Circle
        for slot, motion in self.motions.items():
            x, y = motion.position(time_s)
            self.current[slot] = Circle(x, y, motion.radius_m)
        return tuple(self.current)

    def _check(self, slot):
        if type(slot) is not int or not 0 <= slot < self.plan.capacity:
            raise ValueError("slot outside allocated storage")
