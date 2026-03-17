"""
Shared channel name resolution utilities.

EEG hardware vendors use inconsistent channel naming: "F3", "EEG F3",
"EEG F3-Ref", "f3", "F3." are all the same electrode. These helpers
resolve user-specified channel names against whatever the recording
actually contains, using a five-step strategy:

  1. Exact match
  2. Case-insensitive match
  3. Normalized match (strips trailing dots, "EEG " prefixes, "-Ref" suffixes)
  4. Word-boundary substring (prevents "C3" matching "FC3")

Also provides :func:`detect_naming_convention` for auto-detecting
common channel naming patterns (prefixes, suffixes) on file load.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence


def _normalize_ch_name(name: str) -> str:
    """Strip common EEG channel-name junk to get the core electrode label.

    Handles trailing dots (PhysioNet), leading "EEG "/"EEG_"/"EEG-" prefixes,
    and trailing reference suffixes like "-Ref", "-REF", ".ref".
    """
    s = name.strip()
    # Strip common prefixes (case-insensitive)
    for prefix in ("eeg ", "eeg_", "eeg-", "eeg."):
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
            break
    # Strip common suffixes (case-insensitive)
    for suffix in ("-ref", ".ref", "-ave", "-avg", " ref"):
        if s.lower().endswith(suffix):
            s = s[: -len(suffix)]
            break
    # Strip trailing dots/periods (PhysioNet convention)
    s = s.rstrip(".")
    return s


def resolve_channel(name: str, ch_names: Sequence[str]) -> str:
    """Resolve a user-supplied channel name against the recording's channel list.

    Strategy:
      1. Exact match  →  return immediately
      2. Case-insensitive  →  return the actual channel name
      3. Normalized match  →  strip common EEG prefixes, suffixes, and
         trailing dots from both sides, then compare case-insensitively
      4. Word-boundary substring  →  query must appear at a word boundary
         in the channel name (prevents "C3" matching "FC3")

    Parameters
    ----------
    name : str
        User-specified channel name (e.g. "F3", "Cz", "eeg f3").
    ch_names : sequence of str
        Channel names present in the recording / MNE object.

    Returns
    -------
    str
        The actual channel name as it appears in *ch_names*.

    Raises
    ------
    ValueError
        If no match is found or the name is ambiguous.
    """
    name = name.strip()
    if not name:
        raise ValueError("Channel name cannot be empty.")

    # 1. Exact match
    if name in ch_names:
        return name

    # 2. Case-insensitive
    lower_map = {ch.lower(): ch for ch in ch_names}
    if name.lower() in lower_map:
        return lower_map[name.lower()]

    # 3. Normalized match — strip EEG junk from both sides, compare cores
    query_norm = _normalize_ch_name(name).lower()
    if query_norm:
        norm_matches = [
            ch for ch in ch_names
            if _normalize_ch_name(ch).lower() == query_norm
        ]
        if len(norm_matches) == 1:
            return norm_matches[0]

    # 4. Word-boundary substring (e.g. "F3" matches "EEG F3-Ref" but not "AF3")
    # We use a regex that requires the query NOT be preceded/followed by
    # an alphanumeric character of the same class (letter→letter, digit→digit).
    boundary_pattern = re.compile(
        r"(?<![A-Za-z])" + re.escape(name) + r"(?![A-Za-z])",
        re.IGNORECASE,
    )
    boundary_matches = [ch for ch in ch_names if boundary_pattern.search(ch)]
    if len(boundary_matches) == 1:
        return boundary_matches[0]
    if len(boundary_matches) > 1:
        # Multiple boundary matches — still ambiguous
        raise ValueError(
            f"Channel '{name}' is ambiguous — multiple channels match: "
            f"{boundary_matches[:5]}. Specify the full channel name."
        )

    raise ValueError(
        f"Channel '{name}' not found in recording. "
        f"Available channels (first 10): {list(ch_names)[:10]}"
    )


def resolve_channel_optional(name: str, ch_names: Sequence[str]) -> str | None:
    """Like :func:`resolve_channel` but returns ``None`` instead of raising."""
    try:
        return resolve_channel(name, ch_names)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Standard 10-20 electrode names (for detecting non-standard naming)
# ---------------------------------------------------------------------------

_STANDARD_10_20 = frozenset({
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
    "A1", "A2", "Fpz", "Oz",
    # 10-10 extensions (most common)
    "AF3", "AF4", "AF7", "AF8", "AFz",
    "F1", "F2", "F5", "F6", "F9", "F10",
    "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FCz",
    "FT7", "FT8", "FT9", "FT10",
    "C1", "C2", "C5", "C6",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPz",
    "TP7", "TP8", "TP9", "TP10",
    "P1", "P2", "P5", "P6", "P7", "P8", "P9", "P10", "POz",
    "PO3", "PO4", "PO7", "PO8",
    "T7", "T8", "P7", "P8",  # renamed T3/T4/T5/T6 in 10-10
    "Iz", "I1", "I2",
})


def detect_naming_convention(ch_names: Sequence[str]) -> dict:
    """Analyze channel names and detect common naming patterns.

    Detects:
      - Common prefix shared by most EEG-like channels (e.g. "EEG ", "EEG_")
      - Common suffix shared by most EEG-like channels (e.g. "-Ref", "-REF")
      - Whether names already match standard 10-20 / 10-10 labels

    Returns a dict with:
      - ``detected_prefix`` (str | None): the most common prefix, or None
      - ``detected_suffix`` (str | None): the most common suffix, or None
      - ``standard_match_pct`` (int): percentage of channels matching standard 10-20/10-10
      - ``rename_suggestion`` (str | None): human-readable suggestion
      - ``rename_params`` (dict | None): params for rename_channels node, or None
    """
    if not ch_names:
        return {
            "detected_prefix": None,
            "detected_suffix": None,
            "standard_match_pct": 100,
            "rename_suggestion": None,
            "rename_params": None,
        }

    names = list(ch_names)

    # --- Check standard match (case-insensitive) ---
    standard_lower = {s.lower() for s in _STANDARD_10_20}
    standard_matches = sum(1 for n in names if n.lower() in standard_lower)
    standard_pct = round(100 * standard_matches / len(names)) if names else 100

    # If most channels already match, no renaming needed
    if standard_pct >= 80:
        return {
            "detected_prefix": None,
            "detected_suffix": None,
            "standard_match_pct": standard_pct,
            "rename_suggestion": None,
            "rename_params": None,
        }

    # --- Detect common prefix ---
    detected_prefix = _detect_common_affix(names, mode="prefix")

    # --- Detect common suffix ---
    detected_suffix = _detect_common_affix(names, mode="suffix")

    # --- After stripping detected prefix/suffix, check if names become standard ---
    stripped = names
    if detected_prefix:
        stripped = [n[len(detected_prefix):] if n.startswith(detected_prefix) else n for n in stripped]
    if detected_suffix:
        stripped = [n[:-len(detected_suffix)] if n.endswith(detected_suffix) else n for n in stripped]

    stripped_matches = sum(1 for n in stripped if n.lower() in standard_lower)
    stripped_pct = round(100 * stripped_matches / len(names)) if names else 0

    # Build suggestion
    rename_params: dict | None = None
    suggestion_parts = []

    if detected_prefix and stripped_pct > standard_pct:
        suggestion_parts.append(f'strip prefix "{detected_prefix}"')
        rename_params = rename_params or {}
        rename_params["strip_prefix"] = detected_prefix

    if detected_suffix and stripped_pct > standard_pct:
        suggestion_parts.append(f'strip suffix "{detected_suffix}"')
        rename_params = rename_params or {}
        rename_params["strip_suffix"] = detected_suffix

    rename_suggestion = None
    if suggestion_parts:
        rename_suggestion = (
            f"Channel names have a non-standard naming convention. "
            f"Suggested: {' and '.join(suggestion_parts)} "
            f"to get standard 10-20/10-10 names ({stripped_pct}% match after rename)."
        )

    return {
        "detected_prefix": detected_prefix,
        "detected_suffix": detected_suffix,
        "standard_match_pct": standard_pct,
        "rename_suggestion": rename_suggestion,
        "rename_params": rename_params,
    }


def _detect_common_affix(names: list[str], mode: str = "prefix") -> str | None:
    """Detect a common prefix or suffix shared by >= 60% of channel names.

    Only considers non-alphanumeric-bounded affixes (e.g. "EEG " ends with
    a space/underscore, "-Ref" starts with a dash).
    """
    if len(names) < 2:
        return None

    # Known common patterns to check
    if mode == "prefix":
        candidates = _extract_prefix_candidates(names)
    else:
        candidates = _extract_suffix_candidates(names)

    if not candidates:
        return None

    # Pick the most frequent candidate that appears in >= 60% of names
    counts = Counter(candidates)
    threshold = len(names) * 0.6
    for affix, count in counts.most_common():
        if count >= threshold and len(affix) >= 1:
            return affix

    return None


def _extract_prefix_candidates(names: list[str]) -> list[str]:
    """Extract prefix candidates from channel names."""
    candidates = []
    # Match prefixes like "EEG ", "EEG_", "EEG-"  (letters/digits + separator)
    prefix_re = re.compile(r"^([A-Za-z]+[\s_\-.:]+)")
    for name in names:
        m = prefix_re.match(name)
        if m:
            candidates.append(m.group(1))
    return candidates


def _extract_suffix_candidates(names: list[str]) -> list[str]:
    """Extract suffix candidates from channel names."""
    candidates = []
    # Match suffixes like "-Ref", "-REF", ".eeg", " REF"
    suffix_re = re.compile(r"([\s_\-.:]+[A-Za-z]+\d*)$")
    for name in names:
        m = suffix_re.search(name)
        if m:
            candidates.append(m.group(1))
    return candidates
