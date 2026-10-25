"""Preassigned balanced evaluation populations, independent of finish order."""
from __future__ import annotations


def episode_quota(episodes_requested, num_envs):
    """Return an exact equal quota; never round or truncate an episode budget."""
    for name, value in (("episodes_requested", episodes_requested), ("num_envs", num_envs)):
        if type(value) is not int or value <= 0:
            raise ValueError(name + " must be a positive integer")
    if episodes_requested % num_envs:
        raise ValueError("episodes_requested must be divisible by num_envs")
    return episodes_requested // num_envs


class FixedEpisodeGrid:
    """Accept exactly the first quota episodes from every declared environment.

    Fast slots continue stepping while slow slots finish, but their additional
    completions never enter the assigned population. Incomplete grids must be
    labelled incomplete or rejected, never replaced with faster completions.
    """

    def __init__(self, num_envs, episodes_requested):
        self.quota = episode_quota(episodes_requested, num_envs)
        self.num_envs = num_envs
        self.counts = [0] * num_envs
        self.expected = {(env, ep) for env in range(num_envs) for ep in range(self.quota)}

    @property
    def complete(self):
        return all(count == self.quota for count in self.counts)

    def accept(self, env):
        if type(env) is not int or not 0 <= env < self.num_envs:
            raise ValueError("environment index must belong to the declared grid")
        ep = self.counts[env]
        if ep == self.quota:
            return None
        self.counts[env] += 1
        return ep

    def require_complete(self):
        if not self.complete:
            missing = sorted((env, ep) for env, ep in self.expected if ep >= self.counts[env])
            raise RuntimeError("incomplete evaluation grid; missing keys: " + repr(missing))


def assert_frozen_level(base, level):
    """Reject configuration or runtime curriculum drift during certification."""
    cfg = base.cfg
    if getattr(cfg, "curriculum_frozen", None) is not True:
        raise RuntimeError("evaluation curriculum is not frozen")
    if getattr(cfg, "eval_level", None) != level:
        raise RuntimeError("evaluation level changed during certification")
    for name in ("_curriculum_level", "curriculum_level"):
        actual = getattr(base, name, None)
        if actual is not None:
            equal = actual == level
            if hasattr(equal, "all"):
                equal = equal.all()
            if not bool(equal):
                raise RuntimeError("runtime curriculum level changed during certification")
