#Now to classify processes 

#To do this, two steps are taken 

#First step: Check a rule table, if not found in rule table, Second: Use LLm call the VLM model 

#Importing the important packages
import io 
import os 
import json
import pickle 
from dataclasses import dataclass, field 
from typing import Optional 


from alignment.symbol_parser import GroundedFeature
from manufacturing.rag_retriever import get_retriever 


@dataclass
class MachiningOperation:
    """One machining operation required for one feature."""
    operation_id   : str
    feature_id     : str
    operation_type : str        # drilling | boring | milling | turning | reaming | grinding
    machine_type   : str        # CNC machining centre | CNC lathe | surface grinder
    tooling        : list[str]  # list of tools required
    precision_class: str        # IT grade e.g. IT7
    surface_finish : str        # achievable Ra range e.g. Ra 0.8-1.6 µm
    setup_time_min : float
    cycle_time_min : float
    operator_skill : str        # CNC operator | machinist | grinder specialist
    notes          : Optional[str] = None
    confidence     : float = 1.0   # 1.0 for rule-based, lower for LLM


@dataclass
class ManufacturingReport:
    #It has all the details needed for one draieng 

    #Features list 
    features_processed: int 
    operations: list[MachiningOperation]
    total_setup_min: float 
    total_cycle_min: float 
    total_human_hours: float 
    machines_required: list[str]
    tooling_required: list[str]
    flags: list[str] = field(default_factory=list)



#ISO GRADE LOOKUP


FIT_TO_GRADE = {
    "H5": "IT5", "H6": "IT6", "H7": "IT7", "H8": "IT8",
    "H9": "IT9", "H10": "IT10", "H11": "IT11",
    "g6": "IT6", "h6": "IT6", "k6": "IT6", "n6": "IT6",
    "f7": "IT7", "h7": "IT7",
    "f8": "IT8", "e8": "IT8",
    "d9": "IT9", "c11": "IT11",
}


RULES = [
    # ── Holes ────────────────────────────────────────────────────────────────
    {
        "feature_types"  : ["through_hole", "hole", "blind_hole"],
        "it_grades"      : ["IT11", "IT12", "coarse"],
        "operation_type" : "drilling",
        "machine_type"   : "CNC machining centre",
        "tooling_tmpl"   : ["Twist drill Ø{nom}mm"],
        "precision_class": "IT11",
        "surface_finish" : "Ra 6.3–12.5 µm",
        "setup_min"      : 5.0,
        "cycle_base_min" : 2.0,
        "operator_skill" : "CNC operator",
    },
    {
        "feature_types"  : ["through_hole", "hole", "blind_hole"],
        "it_grades"      : ["IT8", "IT9", "IT10"],
        "operation_type" : "drilling + reaming",
        "machine_type"   : "CNC machining centre",
        "tooling_tmpl"   : ["Twist drill Ø{under}mm", "Machine reamer Ø{nom}mm"],
        "precision_class": "IT8",
        "surface_finish" : "Ra 0.8–1.6 µm",
        "setup_min"      : 8.0,
        "cycle_base_min" : 4.0,
        "operator_skill" : "CNC operator",
    },
    {
        "feature_types"  : ["through_hole", "hole", "blind_hole"],
        "it_grades"      : ["IT6", "IT7"],
        "operation_type" : "drilling + boring + reaming",
        "machine_type"   : "CNC machining centre",
        "tooling_tmpl"   : [
            "Twist drill Ø{under}mm",
            "Boring bar Ø{nom}mm",
            "Precision reamer Ø{nom}mm H7",
        ],
        "precision_class": "IT7",
        "surface_finish" : "Ra 0.4–0.8 µm",
        "setup_min"      : 15.0,
        "cycle_base_min" : 8.0,
        "operator_skill" : "CNC operator",
    },
    {
        "feature_types"  : ["through_hole", "hole", "blind_hole"],
        "it_grades"      : ["IT5", "IT6_fine"],
        "operation_type" : "drilling + boring + grinding",
        "machine_type"   : "CNC machining centre + cylindrical grinder",
        "tooling_tmpl"   : [
            "Twist drill Ø{under}mm",
            "Boring bar Ø{nom}mm",
            "Internal grinding wheel",
        ],
        "precision_class": "IT6",
        "surface_finish" : "Ra 0.2–0.4 µm",
        "setup_min"      : 25.0,
        "cycle_base_min" : 15.0,
        "operator_skill" : "grinder specialist",
    },
    {
        "feature_types"  : ["thread"],
        "it_grades"      : ["any"],
        "operation_type" : "drilling + tapping",
        "machine_type"   : "CNC machining centre",
        "tooling_tmpl"   : [
            "Tap drill Ø{tap_drill}mm",
            "Machine tap M{nom}",
        ],
        "precision_class": "6H",
        "surface_finish" : "Ra 1.6–3.2 µm",
        "setup_min"      : 5.0,
        "cycle_base_min" : 2.5,
        "operator_skill" : "CNC operator",
    },
    {
        "feature_types"  : ["slot", "keyway", "groove"],
        "it_grades"      : ["any"],
        "operation_type" : "end milling",
        "machine_type"   : "CNC machining centre",
        "tooling_tmpl"   : ["End mill Ø{nom}mm", "Slot drill Ø{nom}mm"],
        "precision_class": "IT9",
        "surface_finish" : "Ra 1.6–3.2 µm",
        "setup_min"      : 10.0,
        "cycle_base_min" : 5.0,
        "operator_skill" : "CNC operator",
    },
    {
        "feature_types"  : ["flat_face", "face", "pad", "boss"],
        "it_grades"      : ["any"],
        "operation_type" : "face milling",
        "machine_type"   : "CNC machining centre",
        "tooling_tmpl"   : ["Face mill Ø80mm", "Indexable carbide inserts"],
        "precision_class": "IT9",
        "surface_finish" : "Ra 0.8–3.2 µm",
        "setup_min"      : 8.0,
        "cycle_base_min" : 4.0,
        "operator_skill" : "CNC operator",
    },
    {
        "feature_types"  : ["cylindrical_face", "stepped_face"],
        "it_grades"      : ["IT6", "IT7", "IT8"],
        "operation_type" : "CNC turning",
        "machine_type"   : "CNC lathe",
        "tooling_tmpl"   : [
            "CNMG 120408 turning insert",
            "SDJCR tool holder",
        ],
        "precision_class": "IT7",
        "surface_finish" : "Ra 0.4–1.6 µm",
        "setup_min"      : 12.0,
        "cycle_base_min" : 6.0,
        "operator_skill" : "CNC operator",
    },
    {
        "feature_types"  : ["chamfer", "fillet"],
        "it_grades"      : ["any"],
        "operation_type" : "chamfering",
        "machine_type"   : "CNC machining centre",
        "tooling_tmpl"   : ["Chamfer mill 45°"],
        "precision_class": "IT12",
        "surface_finish" : "Ra 1.6–3.2 µm",
        "setup_min"      : 3.0,
        "cycle_base_min" : 1.0,
        "operator_skill" : "CNC operator",
    },
    {
        "feature_types"  : ["o_ring_groove", "snap_ring_groove"],
        "it_grades"      : ["any"],
        "operation_type" : "grooving",
        "machine_type"   : "CNC lathe",
        "tooling_tmpl"   : ["Grooving insert 3mm", "MGMN 300 parting blade"],
        "precision_class": "IT9",
        "surface_finish" : "Ra 0.8–1.6 µm",
        "setup_min"      : 5.0,
        "cycle_base_min" : 2.0,
        "operator_skill" : "CNC operator",
    },
]


#Classifying all features 
def classify_all(
    grounded_features : list[GroundedFeature],
    vlm_caller        = None,
) -> ManufacturingReport:
    """
    Classify every grounded feature into machining operations.
    Compile totals into a ManufacturingReport.

    Args:
        grounded_features : output of alignment/symbol_parser.py
        vlm_caller        : optional callable for LLM fallback
                            if None, unknown features are flagged only
    """
    all_operations = []
    flags          = []
    op_counter     = [0]

    for feat in grounded_features:
        ops = _classify_one(feat, op_counter, vlm_caller, flags)
        all_operations.extend(ops)
    
    # ── Remove invalid operations from LLM responses ──────────────────────────
    before_count  = len(all_operations)
    all_operations = [
        op for op in all_operations
        if op.setup_time_min  >= 0
        and op.cycle_time_min >= 0
        and op.operation_type.lower().strip() not in ("none", "unknown", "")
        and op.machine_type.strip() != ""
    ]
    removed = before_count - len(all_operations)
    if removed > 0:
        print(f"[Classifier] Removed {removed} invalid operations")
    
    total_setup = sum(op.setup_time_min for op in all_operations)
    total_cycle = sum(op.cycle_time_min for op in all_operations)
    total_hours = (total_setup + total_cycle) / 60.0

    machines = sorted(set(op.machine_type for op in all_operations))
    tooling  = sorted(set(t for op in all_operations for t in op.tooling))

    report = ManufacturingReport(
        features_processed = len(grounded_features),
        operations         = all_operations,
        total_setup_min    = round(total_setup, 1),
        total_cycle_min    = round(total_cycle, 1),
        total_human_hours    = round(total_hours, 2),
        machines_required  = machines,
        tooling_required   = tooling,
        flags              = flags,
    )

    _print_report(report)
    return report

def _classify_one(
    feat       : GroundedFeature,
    op_counter : list[int],
    vlm_caller,
    flags      : list[str],
) -> list[MachiningOperation]:
    """Try rule table first, fall back to LLM if no rule matches."""
    it_grade = _get_it_grade(feat)
    rule     = _match_rule(feat.feature_type, it_grade)

    if rule:
        return [_rule_to_op(rule, feat, it_grade, op_counter)]

    # No rule matched — try LLM + RAG
    if vlm_caller:
        return _llm_classify(feat, it_grade, vlm_caller, op_counter, flags)

    # No LLM available — flag for review
    flags.append(
        f"{feat.feature_id} ({feat.feature_type}): "
        f"no rule matched for {it_grade} — manual review required"
    )
    return []

def _get_it_grade(feat: GroundedFeature) -> str:
    """Extract ISO tolerance grade from feature dimensions."""
    for dim in feat.dimensions:
        if dim.fit_class and dim.fit_class in FIT_TO_GRADE:
            return FIT_TO_GRADE[dim.fit_class]
        if dim.upper_dev_mm is not None and dim.nominal_mm:
            tol = abs(dim.upper_dev_mm) + abs(dim.lower_dev_mm or 0)
            if tol < 0.013:  return "IT6"
            elif tol < 0.021: return "IT7"
            elif tol < 0.033: return "IT8"
            elif tol < 0.052: return "IT9"
            else:             return "IT10"
    return "coarse"

def _match_rule(feature_type: str, it_grade: str) -> Optional[dict]:
    """Find best matching rule for this feature type and tolerance grade."""
    for rule in RULES:
        type_match  = feature_type in rule["feature_types"]
        grade_match = "any" in rule["it_grades"] or it_grade in rule["it_grades"]
        if type_match and grade_match:
            return rule
    return None

def _rule_to_op(
    rule       : dict,
    feat       : GroundedFeature,
    it_grade   : str,
    op_counter : list[int],
) -> MachiningOperation:
    """Convert a matched rule into a MachiningOperation."""
    op_counter[0] += 1

    nom = None 
    for dim in feat.dimensions:
        if dim.nominal_mm and dim.nominal_mm > 0.5:
            nom = dim.nominal_mm
            break
    nom_s = f"{nom:.1f}" if nom else "?"
    und_s = f"{(nom - 0.2):.1f}" if nom else "?"
    tap_s = f"{(nom - 1.5):.1f}" if nom else "?"

    tooling = [
        t.replace("{nom}", nom_s)
         .replace("{under}", und_s)
         .replace("{tap_drill}", tap_s)
        for t in rule["tooling_tmpl"]
    ]

    return MachiningOperation(
        operation_id   = f"op_{op_counter[0]:03d}",
        feature_id     = feat.feature_id,
        operation_type = rule["operation_type"],
        machine_type   = rule["machine_type"],
        tooling        = tooling,
        precision_class= rule["precision_class"],
        surface_finish = rule["surface_finish"],
        setup_time_min = rule["setup_min"],
        cycle_time_min = rule["cycle_base_min"],
        operator_skill = rule["operator_skill"],
        confidence     = 1.0,
    )

def _normalize_machine_name(name: str) -> str:
    """
    Standardize machine names from LLM free-form responses
    into consistent names used throughout the report.

    Why needed:
        LLM returns strings like 'CNC machinning centre' (typo),
        'same CNC machining center used for drilling', or
        'CNC machining centre ord drill press'.
        These all mean the same machine but break deduplication.
    """
    if not name:
        return "CNC machining centre"

    name_lower = name.lower()

    if "lathe" in name_lower:
        return "CNC lathe"
    elif "grind" in name_lower:
        return "CNC grinder"
    elif "hon" in name_lower:
        return "honing machine"
    elif "edm" in name_lower:
        return "EDM machine"
    elif "lathe" in name_lower or "turn" in name_lower:
        return "CNC lathe"
    elif "drill" in name_lower and "centre" not in name_lower and "center" not in name_lower:
        return "CNC machining centre"
    elif any(x in name_lower for x in ["mill", "machining", "centre", "center", "cnc"]):
        return "CNC machining centre"
    else:
        return "CNC machining centre"



def _llm_classify(
    feat       : GroundedFeature,
    it_grade   : str,
    vlm_caller,
    op_counter : list[int],
    flags      : list[str],
) -> list[MachiningOperation]:
    """Use LLM + RAG for features not covered by the rule table."""
    retriever   = get_retriever()
    nom         = feat.dimensions[0].nominal_mm if feat.dimensions else None
    rag_context = retriever.retrieve_features(
        feature_type = feat.feature_type,
        iso_grade    = it_grade,
        nominal_mm   = nom,
    )

    system = f"""
You are a senior manufacturing process engineer.

REFERENCE MATERIAL:
{rag_context}

Given a geometric feature, return the required machining operations as JSON.
Return ONLY this JSON structure:
{{
  "operations": [
    {{
      "operation_type": "<drilling|boring|milling|turning|reaming|grinding|tapping>",
      "machine_type"  : "<specific machine>",
      "tooling"       : ["<tool 1>", "<tool 2>"],
      "precision_class": "<single IT grade only e.g. IT7 or IT11, no other text>",
      "surface_finish" : "<Ra value range only e.g. Ra 0.8-1.6 µm, no other text>",
      "setup_time_min" : <float>,
      "cycle_time_min" : <float>,
      "operator_skill" : "<skill level>",
      "notes"          : "<any note or null>"
    }}
  ],
  "confidence": <float 0.0 to 1.0>
}}


"""
    user = f"""
Feature type    : {feat.feature_type}
Description     : {feat.description}
IT grade        : {it_grade}
Dimensions      : {[f"{d.dim_type} {d.nominal_mm}mm" for d in feat.dimensions]}
GD&T callouts   : {[f"{g.symbol} {g.value_mm}mm" for g in feat.gdt_callouts]}
Surface finish  : {feat.surface_finish}

Determine the required machining operations. Return JSON only.
"""
    try:
        raw  = vlm_caller(system, user, "")
        data = json.loads(raw.strip().lstrip("```json").rstrip("```").strip())
        ops  = []
        conf = float(data.get("confidence", 0.7))

        for op_data in data.get("operations", []):
            op_counter[0] += 1
            ops.append(MachiningOperation(
                operation_id   = f"op_{op_counter[0]:03d}",
                feature_id     = feat.feature_id,
                operation_type = op_data.get("operation_type", "unknown"),
                machine_type   = _normalize_machine_name(op_data.get("machine_type", "")),
                tooling        = op_data.get("tooling", []),
                precision_class= op_data.get("precision_class", it_grade),
                surface_finish = op_data.get("surface_finish", "unknown"),
                setup_time_min = float(op_data.get("setup_time_min", 10.0)),
                cycle_time_min = float(op_data.get("cycle_time_min", 5.0)),
                operator_skill = op_data.get("operator_skill", "CNC operator"),
                notes          = op_data.get("notes"),
                confidence     = conf,
            ))

        if conf < 0.7:
            flags.append(f"{feat.feature_id}: low confidence {conf:.2f} — review recommended")

        return ops

    except Exception as e:
        flags.append(f"{feat.feature_id}: LLM classification failed — {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(report: ManufacturingReport) -> None:
    print(f"\n{'='*60}")
    print(f"MANUFACTURING REPORT")
    print(f"{'='*60}")
    print(f"Features processed : {report.features_processed}")
    print(f"Operations planned : {len(report.operations)}")
    print(f"Total setup time   : {report.total_setup_min} min")
    print(f"Total cycle time   : {report.total_cycle_min} min")
    print(f"Total human-hours    : {report.total_human_hours} hrs")
    print(f"\nMachines required:")
    for m in report.machines_required:
        print(f"  • {m}")
    print(f"\nOperations:")
    for op in report.operations:
        print(f"  {op.operation_id} [{op.feature_id}] "
              f"{op.operation_type} — {op.machine_type}")
        print(f"    Tooling : {', '.join(op.tooling)}")
        print(f"    Grade   : {op.precision_class}  "
              f"Finish: {op.surface_finish}")
        print(f"    Time    : setup={op.setup_time_min}min  "
              f"cycle={op.cycle_time_min}min")
    if report.flags:
        print(f"\nFlags for review:")
        for flag in report.flags:
            print(f"  ⚠ {flag}")
    print(f"{'='*60}")

