"""
alignment/symbol_parser.py
--------------------------
Solves Alignment Problem 2.

Converts raw annotation strings into structured typed objects.
Uses rule-based regex parsing — fast, deterministic, no model call.

Three parsers:
    parse_dimension()      — handles linear, diameter, radius values
    parse_gdt()            — handles GD&T feature control frames
    parse_surface_finish() — handles Ra, Rz, Rq surface finish callouts

One orchestrator:
    parse_annotation()     — decides which parser to call based on type
    parse_all()            — runs parse_annotation on every linked annotation
                             and returns GroundedFeature objects
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES — structured output objects
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParsedDimension:
    """
    A parsed linear, diameter, or radius dimension.

    Examples:
        "Φ60"        → nominal_mm=60.0, dim_type="diameter"
        "R1"         → nominal_mm=1.0,  dim_type="radius"
        "21.5"       → nominal_mm=21.5, dim_type="linear"
        "Ø25H7"      → nominal_mm=25.0, dim_type="diameter",
                        fit_class="H7", iso_grade="IT7"
        "76±0.1"     → nominal_mm=76.0, upper_dev=0.1, lower_dev=-0.1
        "+0.02/-0.01"→ upper_dev=0.02, lower_dev=-0.01
    """
    raw          : str
    dim_type     : str              # "linear" | "diameter" | "radius"
    nominal_mm   : Optional[float] = None
    upper_dev_mm : Optional[float] = None
    lower_dev_mm : Optional[float] = None
    fit_class    : Optional[str]   = None   # e.g. "H7", "g6"
    iso_grade    : Optional[str]   = None   # e.g. "IT7"


@dataclass
class ParsedGDT:
    """
    A parsed GD&T feature control frame.

    Examples:
        "⊥ 0.05 A"  → symbol="perpendicularity", value_mm=0.05, datum="A"
        "// 0.1 B"  → symbol="parallelism",       value_mm=0.1,  datum="B"
        "○ 0.02"    → symbol="circularity",        value_mm=0.02, datum=None
    """
    raw       : str
    symbol    : str
    value_mm  : float
    datum     : Optional[str] = None


@dataclass
class ParsedSurfaceFinish:
    """
    A parsed surface finish callout.

    Examples:
        "Ra 1.6"  → parameter="Ra", value_um=1.6
        "Rz 6.3"  → parameter="Rz", value_um=6.3
        "√"       → parameter="Ra", value_um=3.2  (unspecified, default)
    """
    raw        : str
    parameter  : str    # "Ra" | "Rz" | "Rq"
    value_um   : float


@dataclass
class GroundedFeature:
    """
    A geometric feature with all its annotations parsed and linked.
    This is the final output of the alignment stage.
    Fed directly into manufacturing/process_classifier.py.
    """
    feature_id     : str
    feature_type   : str
    description    : str
    location       : dict                           # { cx, cy }

    dimensions     : list[ParsedDimension] = field(default_factory=list)
    gdt_callouts   : list[ParsedGDT]       = field(default_factory=list)
    surface_finish : Optional[ParsedSurfaceFinish] = None
    raw_notes      : list[str]             = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# LOOKUP TABLES
# ─────────────────────────────────────────────────────────────────────────────

# ISO 286 fit class → ISO tolerance grade
# Uppercase = holes, lowercase = shafts
FIT_TO_ISO_GRADE = {
    "H5": "IT5", "H6": "IT6", "H7": "IT7", "H8": "IT8",
    "H9": "IT9", "H10": "IT10", "H11": "IT11", "H12": "IT12",
    "JS7": "IT7", "K7": "IT7", "M7": "IT7", "N7": "IT7", "P7": "IT7",
    "g4": "IT4", "g5": "IT5", "g6": "IT6",
    "h5": "IT5", "h6": "IT6", "h7": "IT7", "h8": "IT8",
    "f6": "IT6", "f7": "IT7", "f8": "IT8",
    "e7": "IT7", "e8": "IT8", "e9": "IT9",
    "d8": "IT8", "d9": "IT9", "d10": "IT10",
    "c11": "IT11", "b11": "IT11", "a11": "IT11",
    "k5": "IT5", "k6": "IT6",
    "m5": "IT5", "m6": "IT6",
    "n5": "IT5", "n6": "IT6",
    "p6": "IT6", "r6": "IT6", "s6": "IT6",
}

# GD&T symbol → canonical name
# Covers Unicode symbols and text alternatives OCR might produce
GDT_SYMBOLS = {
    "⊥"    : "perpendicularity",
    "//"   : "parallelism",
    "∠"    : "angularity",
    "○"    : "circularity",
    "◎"    : "concentricity",
    "⌭"    : "cylindricity",
    "⌒"    : "profile_of_a_line",
    "⌓"    : "profile_of_a_surface",
    "↗"    : "circular_runout",
    "⇗"    : "total_runout",
    "⊙"    : "symmetry",
    "⌀"    : "position",
    "—"    : "flatness",
    # Text alternatives when OCR misreads symbols
    "PERP" : "perpendicularity",
    "PAR"  : "parallelism",
    "FLAT" : "flatness",
    "CYL"  : "cylindricity",
    "CIRC" : "circularity",
    "POS"  : "position",
    "SYM"  : "symmetry",
    "RUN"  : "circular_runout",
}

# Unspecified surface finish marks → Ra equivalent
FINISH_MARKS = {
    "√"    : 3.2,
    "▽"    : 6.3,
    "▽▽"   : 1.6,
    "▽▽▽"  : 0.4,
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def parse_all(
    features    : list[dict],
    annotations : list[dict],
) -> list[GroundedFeature]:
    """
    Main entry point.

    Takes:
        features    : feature dicts from InternVL2
        annotations : annotation dicts from Qwen2.5-VL
                      with linked_feature_id set by spatial_aligner

    Returns:
        List of GroundedFeature objects — one per geometric feature.
        Each GroundedFeature contains all its parsed annotations.

    Steps:
        1. Group annotations by which feature they are linked to
        2. For each feature, parse all its linked annotations
        3. Sort parsed objects into dimensions, GD&T, surface finish, notes
        4. Build and return GroundedFeature objects
    """

    # Step 1 — group annotations by feature
    # { feature_id: [annotation, annotation, ...] }
    ann_by_feature: dict[str, list[dict]] = {
        f["feature_id"]: [] for f in features
    }
    unlinked_annotations = []

    for ann in annotations:
        fid = ann.get("linked_feature_id")
        if fid and fid in ann_by_feature:
            ann_by_feature[fid].append(ann)
        else:
            unlinked_annotations.append(ann)

    if unlinked_annotations:
        print(f"[Parser] {len(unlinked_annotations)} annotations not linked to any feature (general notes)")

    # Step 2 — build a GroundedFeature for each geometric feature
    grounded_features = []

    for feat in features:
        fid        = feat["feature_id"]
        linked_anns = ann_by_feature.get(fid, [])

        dimensions     = []
        gdt_callouts   = []
        surface_finish = None
        raw_notes      = []

        # Step 3 — parse each linked annotation
        for ann in linked_anns:
            ann_type = ann.get("annotation_type", "")
            raw_text = ann.get("raw_text", "")

            if ann_type in (
                "linear_dimension",
                "diameter_dimension",
                "radius_dimension",
                "tolerance",
            ):
                parsed = parse_dimension(raw_text, ann_type)
                if parsed:
                    dimensions.append(parsed)

            elif ann_type == "gdt_callout":
                parsed = parse_gdt(raw_text)
                if parsed:
                    gdt_callouts.append(parsed)

            elif ann_type == "surface_finish":
                parsed = parse_surface_finish(raw_text)
                if parsed:
                    surface_finish = parsed   # keep last if multiple

            elif ann_type == "general_note":
                if raw_text and raw_text != "ILLEGIBLE":
                    raw_notes.append(raw_text)

        # Step 4 — build GroundedFeature
        gf = GroundedFeature(
            feature_id     = fid,
            feature_type   = feat.get("feature_type", "unknown"),
            description    = feat.get("description", ""),
            location       = feat.get("location", {"cx": 0.5, "cy": 0.5}),
            dimensions     = dimensions,
            gdt_callouts   = gdt_callouts,
            surface_finish = surface_finish,
            raw_notes      = raw_notes,
        )
        grounded_features.append(gf)

    _print_summary(grounded_features)
    return grounded_features


# ─────────────────────────────────────────────────────────────────────────────
# PARSER 1 — DIMENSIONS
# ─────────────────────────────────────────────────────────────────────────────

def parse_dimension(raw: str, ann_type: str) -> Optional[ParsedDimension]:
    """
    Parse a dimension or tolerance string.

    Handles these patterns:
        Φ60       — diameter, no tolerance
        Φ60H7     — diameter with fit class
        R1        — radius
        21.5      — linear
        76±0.1    — linear with symmetric tolerance
        25+0.02/-0.01 — asymmetric tolerance
        Ø25H7/g6  — clearance fit pair (takes hole spec)
    """
    raw     = raw.strip()
    cleaned = raw

    # ── Determine dimension type from prefix ──────────────────────────────────
    if re.match(r"^[ΦØ⌀Ø]", cleaned, re.IGNORECASE):
        dim_type = "diameter"
        cleaned  = re.sub(r"^[ΦØ⌀Ø]", "", cleaned).strip()
    elif re.match(r"^R\s*\d", cleaned, re.IGNORECASE):
        dim_type = "radius"
        cleaned  = re.sub(r"^R\s*", "", cleaned, flags=re.IGNORECASE).strip()
    elif ann_type == "diameter_dimension":
        dim_type = "diameter"
    elif ann_type == "radius_dimension":
        dim_type = "radius"
    else:
        dim_type = "linear"

    # ── Extract fit class if present e.g. H7, g6, H7/g6 ─────────────────────
    fit_match = re.search(r"([A-Za-z][S]?\d+)(?:/([A-Za-z][S]?\d+))?", cleaned)
    fit_class = None
    iso_grade = None
    if fit_match:
        fit_class = fit_match.group(1)         # take hole/first spec
        iso_grade = FIT_TO_ISO_GRADE.get(fit_class)

    # ── Extract nominal value ─────────────────────────────────────────────────
    nominal_match = re.match(r"([\d.]+)", cleaned)
    nominal = float(nominal_match.group(1)) if nominal_match else None

    # ── Extract tolerance ─────────────────────────────────────────────────────
    upper_dev = None
    lower_dev = None

    # Symmetric: ±0.1 or ± 0.1
    sym_match = re.search(r"[±]\s*([\d.]+)", cleaned)
    if sym_match:
        tol       = float(sym_match.group(1))
        upper_dev = tol
        lower_dev = -tol

    # Asymmetric: +0.02/-0.01 or +0.02-0.01
    asym_match = re.search(r"\+([\d.]+)\s*/?\s*-\s*([\d.]+)", cleaned)
    if asym_match:
        upper_dev = float(asym_match.group(1))
        lower_dev = -float(asym_match.group(2))

    if nominal is None and fit_class is None:
        return None

    return ParsedDimension(
        raw          = raw,
        dim_type     = dim_type,
        nominal_mm   = nominal,
        upper_dev_mm = upper_dev,
        lower_dev_mm = lower_dev,
        fit_class    = fit_class,
        iso_grade    = iso_grade,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PARSER 2 — GD&T
# ─────────────────────────────────────────────────────────────────────────────

def parse_gdt(raw: str) -> Optional[ParsedGDT]:
    """
    Parse a GD&T feature control frame string.

    Examples:
        "⊥ 0.05 A"   → perpendicularity 0.05 datum A
        "// 0.1 B C" → parallelism 0.1 datum B
        "○ 0.02"     → circularity 0.02 no datum
        "PERP 0.05 A"→ perpendicularity 0.05 datum A (OCR fallback)

    Steps:
        1. Identify the GD&T symbol at the start of the string
        2. Extract the tolerance value (first number after symbol)
        3. Extract datum reference (capital letter after number)
    """
    raw_stripped = raw.strip()

    # Step 1 — identify symbol
    symbol_name = None
    remainder   = raw_stripped

    for sym, name in GDT_SYMBOLS.items():
        if raw_stripped.startswith(sym):
            symbol_name = name
            remainder   = raw_stripped[len(sym):].strip()
            break

    if symbol_name is None:
        return None

    # Step 2 — extract tolerance value
    tol_match = re.search(r"([\d.]+)", remainder)
    if not tol_match:
        return None
    tol_value = float(tol_match.group(1))

    # Step 3 — extract datum reference
    # Datum is a capital letter after the tolerance value
    datum_match = re.search(r"[\d.]+\s+([A-Z](?:\s+[A-Z])*)", remainder)
    datum = datum_match.group(1).strip() if datum_match else None

    return ParsedGDT(
        raw      = raw,
        symbol   = symbol_name,
        value_mm = tol_value,
        datum    = datum,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PARSER 3 — SURFACE FINISH
# ─────────────────────────────────────────────────────────────────────────────

def parse_surface_finish(raw: str) -> Optional[ParsedSurfaceFinish]:
    """
    Parse surface finish callouts.

    Examples:
        "Ra 1.6"  → Ra, 1.6 µm
        "Rz 6.3"  → Rz, 6.3 µm
        "Ra1.6"   → Ra, 1.6 µm  (no space)
        "√"       → Ra, 3.2 µm  (unspecified — machined default)
        "▽▽"      → Ra, 1.6 µm  (triangle notation)
    """
    raw = raw.strip()

    # Ra / Rz / Rq with numeric value
    match = re.match(r"(Ra|Rz|Rq)\s*([\d.]+)", raw, re.IGNORECASE)
    if match:
        return ParsedSurfaceFinish(
            raw       = raw,
            parameter = match.group(1).capitalize(),
            value_um  = float(match.group(2)),
        )

    # Triangle / check mark notation
    if raw in FINISH_MARKS:
        return ParsedSurfaceFinish(
            raw       = raw,
            parameter = "Ra",
            value_um  = FINISH_MARKS[raw],
        )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(grounded_features: list[GroundedFeature]) -> None:
    print(f"\n[Parser] Grounded features: {len(grounded_features)}")
    for gf in grounded_features:
        print(f"\n  {gf.feature_id} — {gf.feature_type} — {gf.description}")
        for d in gf.dimensions:
            tol = ""
            if d.upper_dev_mm is not None:
                tol = f" +{d.upper_dev_mm}/{d.lower_dev_mm}"
            fit = f" [{d.fit_class} {d.iso_grade}]" if d.fit_class else ""
            print(f"    DIM: {d.dim_type} {d.nominal_mm}mm{tol}{fit}")
        for g in gf.gdt_callouts:
            print(f"    GDT: {g.symbol} {g.value_mm}mm datum={g.datum}")
        if gf.surface_finish:
            sf = gf.surface_finish
            print(f"    FINISH: {sf.parameter} {sf.value_um}µm")
        for note in gf.raw_notes:
            print(f"    NOTE: {note}")