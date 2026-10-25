"""Persistent body-frame wrench for the journal envs without the per-call deprecation (G6).

Isaac Lab 2.3.2 ``RigidObject.set_external_force_and_torque(forces, torques)``
(isaaclab/assets/rigid_object/rigid_object.py:402-468) logs a deprecation warning on EVERY
call (once per physics substep here) and then does exactly

    self._permanent_wrench_composer.set_forces_and_torques(
        forces=wp.from_torch(forces, dtype=wp.vec3f), torques=wp.from_torch(torques, dtype=wp.vec3f),
        positions=None, body_ids=self._ALL_BODY_INDICES_WP, env_ids=self._ALL_INDICES_WP,
        is_global=False)

``permanent_wrench_composer.set_forces_and_torques(forces=forces, torques=torques)``
(isaaclab/utils/wrench_composer.py:221-313) resolves env_ids=None / body_ids=None to its
own arange(num_envs) / arange(num_bodies) int32 index arrays (wrench_composer.py:53-58,
built from asset.num_instances / num_bodies, the same counts as rigid_object.py:544-548),
converts torch forces/torques with the same wp.from_torch(..., wp.vec3f), passes empty
positions and is_global=False: the same composed wrench, without the warning. The stream
handler rate-limits that warning to one line per 5 s (isaaclab/utils/logger.py:76, :144-176)
but the file handler (logger.py:88-95, no filter) writes every call to the Isaac Lab log.
Objects without a permanent_wrench_composer (the CPU test fakes, older Isaac Lab) keep the
original call.
"""


def apply_body_wrench(robot, forces, torques):
    """Set the persistent link-frame wrench of every env and body (see the module docstring)."""
    composer = getattr(robot, "permanent_wrench_composer", None)
    if composer is None:
        robot.set_external_force_and_torque(forces, torques)
        return
    composer.set_forces_and_torques(forces=forces, torques=torques)
