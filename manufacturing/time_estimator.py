"""
manufacturing/time_estimator.py
--------------------------------
Refines time estimates using machining formulas.

The process_classifier gives base time estimates from the rule table.
This module applies actual cutting speed and feed rate formulas
to produce more accurate cycle time estimates when nominal
dimension data is available.

Formulas used:
    Drilling:
        RPM = (Vc × 1000) / (π × D)
        cycle_time = (depth / (feed × RPM)) minutes

    Turning:
        RPM = (Vc × 1000) / (π × D)
        cycle_time = (length / (feed × RPM)) minutes

    Milling:
        RPM = (Vc × 1000) / (π × D_tool)
        cycle_time = (length / (fz × z × RPM)) minutes
"""

import math
from manufacturing.process_classifier import MachiningOperation
from alignment.symbol_parser import GroundedFeature
from typing import Optional


# ── Cutting data for steel (medium carbon, HB 200) ────────────────────────────
# Vc = cutting speed in m/min, f = feed in mm/rev or mm/tooth
CUTTING_DATA = {
    "drilling" : {"Vc": 25.0,  "f": 0.15},   # HSS drill, steel
    "reaming"  : {"Vc": 8.0,   "f": 0.20},   # HSS reamer
    "boring"   : {"Vc": 120.0, "f": 0.10},   # carbide boring bar
    "turning"  : {"Vc": 200.0, "f": 0.15},   # carbide insert, finishing
    "milling"  : {"Vc": 120.0, "fz": 0.10, "z": 4},  # 4-flute end mill
    "grinding" : {"Vc": 30.0,  "f": 0.005},  # surface speed, infeed
    "tapping"  : {"Vc": 10.0,  "f": 1.5},    # pitch = feed for tapping
    "chamfering": {"Vc": 100.0, "f": 0.10},
    "grooving" : {"Vc": 80.0,  "f": 0.05},
}

# Assumed depth/length when not available from drawing
DEFAULT_DEPTH_MM  = 25.0
DEFAULT_LENGTH_MM = 50.0


def refine_times(
    operations        : list[MachiningOperation],
    grounded_features : list[GroundedFeature],
) -> list[MachiningOperation]:
    """
    Refine cycle time estimates using machining formulas where possible.

    For each operation, look up the nominal dimension of the linked
    feature and apply the appropriate cutting time formula.
    Falls back to the classifier's base estimate if data is missing.

    Returns the same operations list with updated cycle_time_min values.
    """
    feat_map = {f.feature_id: f for f in grounded_features}

    for op in operations:
        feat = feat_map.get(op.feature_id)
        if not feat or not feat.dimensions:
            continue

        nominal = feat.dimensions[0].nominal_mm
        if not nominal:
            continue

        refined = _calculate_time(op.operation_type, nominal)
        if refined is not None:
            op.cycle_time_min = round(refined, 2)
            if op.notes:
                op.notes += " | time formula applied"
            else:
                op.notes = "time formula applied"

    return operations


def _calculate_time(operation_type: str, nominal_mm: float) -> Optional[float]:
    """
    Apply the correct formula for the operation type.
    Returns cycle time in minutes, or None if formula cannot be applied.
    """
    op_lower = operation_type.lower()

    # Find matching cutting data
    data = None
    for key in CUTTING_DATA:
        if key in op_lower:
            data = CUTTING_DATA[key]
            break

    if data is None:
        return None

    try:
        if "drill" in op_lower:
            return _drilling_time(nominal_mm, data)
        elif "ream" in op_lower:
            return _drilling_time(nominal_mm, data)   # same formula
        elif "bor" in op_lower:
            return _drilling_time(nominal_mm, data)
        elif "turn" in op_lower:
            return _turning_time(nominal_mm, data)
        elif "mill" in op_lower:
            return _milling_time(nominal_mm, data)
        elif "tap" in op_lower:
            return _tapping_time(nominal_mm, data)
        else:
            return None

    except (ZeroDivisionError, ValueError):
        return None


def _drilling_time(diameter_mm: float, data: dict) -> float:
    """
    Drilling / boring / reaming cycle time.

    RPM = (Vc × 1000) / (π × D)
    time = depth / (f × RPM)   [minutes]
    """
    vc    = data["Vc"]
    f     = data["f"]
    depth = DEFAULT_DEPTH_MM

    rpm  = (vc * 1000) / (math.pi * diameter_mm)
    time = depth / (f * rpm)
    return time


def _turning_time(diameter_mm: float, data: dict) -> float:
    """
    CNC turning cycle time.

    RPM = (Vc × 1000) / (π × D)
    time = L / (f × RPM)   [minutes]
    """
    vc     = data["Vc"]
    f      = data["f"]
    length = DEFAULT_LENGTH_MM

    rpm  = (vc * 1000) / (math.pi * diameter_mm)
    time = length / (f * rpm)
    return time


def _milling_time(width_mm: float, data: dict) -> float:
    """
    Milling cycle time.

    RPM = (Vc × 1000) / (π × D_tool)
    vf = fz × z × RPM  [mm/min feed rate]
    time = L / vf
    """
    vc     = data["Vc"]
    fz     = data["fz"]
    z      = data["z"]
    length = DEFAULT_LENGTH_MM
    d_tool = max(width_mm, 6.0)   # tool diameter at least 6mm

    rpm  = (vc * 1000) / (math.pi * d_tool)
    vf   = fz * z * rpm
    time = length / vf
    return time


def _tapping_time(diameter_mm: float, data: dict) -> float:
    """
    Tapping cycle time.
    For tapping, feed = pitch (mm/rev).
    """
    vc    = data["Vc"]
    pitch = data["f"]
    depth = DEFAULT_DEPTH_MM

    rpm  = (vc * 1000) / (math.pi * diameter_mm)
    time = depth / (pitch * rpm)
    return time * 2    # × 2 for retract stroke