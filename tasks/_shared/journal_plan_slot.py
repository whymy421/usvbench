"""JournalPlanSlot: ``cfg.journal_plan`` kept OUT of the Isaac Lab dict utilities (G6).

The 24 journal templates (journal_runtime_cfg_v2) list it as their LAST base, after the
family parent, so the class layout seen by the existing template tests is unchanged and
attribute lookup reaches the property only because no parent defines ``journal_plan``.
No Isaac Lab import: plain Python.
"""


class JournalPlanSlot:
    """``cfg.journal_plan`` kept OUT of the Isaac Lab dict utilities (G6).

    The frozen plan (journal_cfg_v2.JournalTaskPlan) is stored under the dunder key
    PLAN_KEY of the instance ``__dict__``. Isaac Lab 2.3.2 skips "__" names in
    class_to_dict (dict.py:57-60), _validate (configclass.py:286-289) and the
    configclass post-init (configclass.py:392-395); hydra's from_dict therefore never
    receives the plan and never assigns into the frozen dataclass (the P5 failure
    ``FrozenInstanceError: cannot assign to field 'geometry_tier'`` at dict.py:163,
    reached from hydra.py:90). The plan object is unchanged: same class, built once
    in _journal_post_init from (journal_task, vehicle, journal_geometry_tier), and
    copy.copy / copy.deepcopy / pickle carry it with ``__dict__``. Unset reads None
    (``getattr(cfg, "journal_plan", None)`` as before); the configclass post-init's
    deepcopy re-assignment of every dir() member (configclass.py:401-402) goes through
    the setter exactly as it re-assigned the plain attribute before.
    """

    @property
    def journal_plan(self):
        return self.__dict__.get(PLAN_KEY)

    @journal_plan.setter
    def journal_plan(self, plan):
        self.__dict__[PLAN_KEY] = plan


PLAN_KEY = "__journal_plan__"
# cert_v3.json_value renders hidden cfg members under their public names, at their
# original position, exactly as before (resolved_config unchanged).
HIDDEN_CFG_MEMBERS = {PLAN_KEY: "journal_plan"}
