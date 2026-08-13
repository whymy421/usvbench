"""Compatibility imports for the shared JONSWAP implementation."""

from .._shared.legacy_jonswap import (
    JONSWAPWaveCfg,
    LegacyJONSWAPWaveField as JONSWAPWaveField,
    validate_spectrum,
)

__all__ = ["JONSWAPWaveCfg", "JONSWAPWaveField", "validate_spectrum"]
