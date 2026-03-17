"""
backend/api/report_routes.py

POST /pipeline/report -- generate a PDF clinical report from pipeline results.

This is a post-execution endpoint, NOT a DAG node. After the frontend calls
POST /pipeline/execute and displays results, the user can call this endpoint
to bundle all metrics and plots into a downloadable PDF.

The endpoint scans node_results for:
  - metrics entries (output_type == "dict", i.e., nodes returning dict outputs)
  - plot entries (data field containing a base64 PNG data URI)

Enhanced (Tier A):
  - Data Quality summary (from session_info)
  - Pipeline Configuration (node list with params)
  - Clinical Interpretation (reference ranges for known metrics)
  - Audit Trail (parameter change history)
  - Clinician Notes

PDF is returned as a binary application/pdf response.
"""

from __future__ import annotations

import base64
import datetime
import io
from typing import Any, Optional

from fastapi import APIRouter
from fastapi.responses import Response

from backend.models import ReportRequest, ReportSections

try:
    from fpdf import FPDF
    _FPDF_AVAILABLE = True
except ImportError:
    _FPDF_AVAILABLE = False

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

# Unicode -> latin-1 safe replacements for fpdf2 built-in fonts.
_UNICODE_REPLACEMENTS = {
    "\u2013": "-",   # en-dash -> hyphen
    "\u2014": "--",  # em-dash -> double hyphen
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u2026": "...", # ellipsis
    "\u00b5": "u",   # micro sign -> u
    "\u2192": "->",  # right arrow
    "\u2190": "<-",  # left arrow
    "\u2265": ">=",  # greater-than-or-equal
    "\u2264": "<=",  # less-than-or-equal
}


def _sanitize_text(text: str) -> str:
    """Replace non-latin-1 characters so fpdf2 Helvetica doesn't crash."""
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    # Drop any remaining non-latin-1 characters
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ---------------------------------------------------------------------------
# Clinical interpretation reference ranges
# ---------------------------------------------------------------------------

# Maps node_type -> metric_key -> (label, normal_range_str, interpret_fn)
# interpret_fn(value) -> ("normal"|"borderline"|"abnormal", explanation)
_CLINICAL_REFERENCES: dict[str, dict[str, tuple[str, str, Any]]] = {
    "compute_alpha_peak": {
        "iaf_hz": (
            "Individual Alpha Frequency",
            "8.0 - 13.0 Hz (healthy adult)",
            lambda v: (
                ("normal", "Within healthy adult range")
                if 8.0 <= v <= 13.0
                else ("borderline", "Slightly outside typical range (8-13 Hz)")
                if 7.0 <= v <= 14.0
                else ("abnormal", "Outside normal range; consider clinical review")
            ),
        ),
    },
    "compute_band_ratio": {
        "band_ratio": (
            "Theta/Beta Ratio",
            "Linear: 1.5 - 3.0 (healthy adult); Log10: varies",
            lambda v: (
                # This interpretation applies to linear TBR; log10 varies
                ("info", "Interpret in context of scale (log10 vs linear) and age norms")
            ),
        ),
    },
    "compute_asymmetry": {
        "asymmetry_index": (
            "Frontal Alpha Asymmetry",
            "-0.5 to +0.5 (typical range)",
            lambda v: (
                ("normal", "Within typical range")
                if -0.5 <= v <= 0.5
                else ("borderline", "Moderate asymmetry; may indicate lateralized processing")
                if -1.0 <= v <= 1.0
                else ("abnormal", "Marked asymmetry; clinical review recommended")
            ),
        ),
    },
    "detect_spikes": {
        "n_spikes": (
            "Spike Count",
            "0 events (healthy adult)",
            lambda v: (
                ("normal", "No candidate spikes detected")
                if v == 0
                else ("borderline", f"{v} candidate spike(s) detected; clinical confirmation required")
                if v <= 10
                else ("abnormal", f"{v} candidate spikes detected; further clinical review recommended")
            ),
        ),
    },
}

# Color coding for PDF (R, G, B)
_INTERPRETATION_COLORS = {
    "normal": (34, 139, 34),      # forest green
    "borderline": (200, 150, 0),  # amber
    "abnormal": (200, 50, 50),    # red
    "info": (80, 80, 160),        # blue-grey
}


def _format_value(value: Any) -> str:
    """Format a metric value for display."""
    if isinstance(value, list):
        val_str = ", ".join(str(v) for v in value[:15])
        if len(value) > 15:
            val_str += f" ... (+{len(value) - 15} more)"
        return val_str
    elif isinstance(value, float):
        return f"{value:.4f}"
    else:
        return str(value)


def _draw_hr(pdf: "FPDF") -> None:
    """Draw a horizontal rule."""
    pdf.set_draw_color(180, 180, 180)
    pdf.set_line_width(0.3)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)


def _draw_section_heading(pdf: "FPDF", title: str) -> None:
    """Draw a section heading."""
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, _sanitize_text(title), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def _draw_kv_row(pdf: "FPDF", label: str, value: str, label_width: int = 65) -> None:
    """Draw a key-value row."""
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(label_width, 6, _sanitize_text(f"  {label}"))
    pdf.set_font("Helvetica", "", 9)
    x_after_label = pdf.get_x()
    y_before = pdf.get_y()
    pdf.multi_cell(
        pdf.w - pdf.r_margin - x_after_label,
        6,
        _sanitize_text(value),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    if pdf.get_y() == y_before:
        pdf.ln(6)


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def _generate_pdf(
    node_results: dict,
    title: str,
    patient_id: str,
    clinic_name: str,
    session_info: Optional[dict] = None,
    pipeline_config: Optional[list[dict]] = None,
    audit_log: Optional[list[dict]] = None,
    notes: str = "",
    sections: Optional[ReportSections] = None,
    included_nodes: Optional[list[str]] = None,
) -> bytes:
    """
    Builds a PDF report from node_results and optional context.

    Sections (all toggleable):
      1. Header -- title, date, patient info
      2. Data Quality -- recording metadata (sfreq, channels, duration)
      3. Pipeline Configuration -- processing steps with parameters
      4. Analysis Results -- metrics with clinical interpretation
      5. Visualizations -- embedded PNG plots
      6. Audit Trail -- parameter change history
      7. Notes -- free-text clinician observations

    If *included_nodes* is provided, only those node IDs appear in the
    Analysis Results and Visualizations sections.

    Returns the PDF as a bytes object.
    """
    if sections is None:
        sections = ReportSections()

    # Filter to user-selected nodes when specified
    if included_nodes is not None:
        include_set = set(included_nodes)
        node_results = {
            nid: entry for nid, entry in node_results.items()
            if nid in include_set
        }

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # === Header (always shown) ===
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, _sanitize_text(title), new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(0, 6, f"Generated: {ts}", new_x="LMARGIN", new_y="NEXT", align="C")
    if patient_id:
        pdf.cell(0, 6, _sanitize_text(f"Patient ID: {patient_id}"), new_x="LMARGIN", new_y="NEXT", align="C")
    if clinic_name:
        pdf.cell(0, 6, _sanitize_text(f"Clinic: {clinic_name}"), new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)
    _draw_hr(pdf)

    has_content = False

    # === Data Quality Summary ===
    if sections.data_quality and session_info:
        has_content = True
        _draw_section_heading(pdf, "Data Quality Summary")

        sfreq = session_info.get("sfreq", "N/A")
        nchan = session_info.get("nchan", "N/A")
        duration_s = session_info.get("duration_s", 0)
        if isinstance(duration_s, (int, float)) and duration_s > 0:
            mins = int(duration_s // 60)
            secs = duration_s % 60
            duration_str = f"{mins}m {secs:.1f}s" if mins > 0 else f"{secs:.1f}s"
        else:
            duration_str = "N/A"

        _draw_kv_row(pdf, "Sampling Rate", f"{sfreq} Hz")
        _draw_kv_row(pdf, "Channels", str(nchan))
        _draw_kv_row(pdf, "Duration", duration_str)

        ch_types = session_info.get("ch_types", {})
        if ch_types:
            types_str = ", ".join(f"{t}: {c}" for t, c in ch_types.items())
            _draw_kv_row(pdf, "Channel Types", types_str)

        bads = session_info.get("bads", [])
        if bads:
            _draw_kv_row(pdf, "Bad Channels", ", ".join(bads))
        else:
            _draw_kv_row(pdf, "Bad Channels", "None marked")

        highpass = session_info.get("highpass", 0)
        lowpass = session_info.get("lowpass", 0)
        if highpass or lowpass:
            _draw_kv_row(pdf, "Hardware Filters", f"HP: {highpass} Hz, LP: {lowpass} Hz")

        meas_date = session_info.get("meas_date")
        if meas_date:
            _draw_kv_row(pdf, "Recording Date", str(meas_date))

        n_annot = session_info.get("n_annotations", 0)
        if n_annot > 0:
            labels = session_info.get("annotation_labels", [])
            _draw_kv_row(pdf, "Annotations", f"{n_annot} events ({', '.join(labels[:10])})")

        pdf.ln(3)
        _draw_hr(pdf)

    # === Pipeline Configuration ===
    if sections.pipeline_config and pipeline_config:
        has_content = True
        _draw_section_heading(pdf, "Pipeline Configuration")

        for i, step in enumerate(pipeline_config, 1):
            node_type = step.get("node_type", "unknown")
            label = step.get("label", node_type)
            params = step.get("parameters", {})

            # Step heading
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(240, 240, 248)
            pdf.cell(
                0, 7,
                _sanitize_text(f"  Step {i}: {label} ({node_type})"),
                fill=True, new_x="LMARGIN", new_y="NEXT",
            )

            # Parameters
            if params:
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(80, 80, 80)
                param_strs = []
                for k, v in params.items():
                    if k == "file_path":
                        continue  # Skip file paths
                    param_strs.append(f"{k}={v}")
                if param_strs:
                    params_text = "    " + ", ".join(param_strs)
                    pdf.multi_cell(
                        0, 5, _sanitize_text(params_text),
                        new_x="LMARGIN", new_y="NEXT",
                    )
                pdf.set_text_color(0, 0, 0)

            pdf.ln(1)

        pdf.ln(2)
        _draw_hr(pdf)

    # === Analysis Results (with Clinical Interpretation) ===
    metrics_entries = {
        node_id: entry
        for node_id, entry in node_results.items()
        if (
            entry.get("status") == "success"
            and isinstance(entry.get("metrics"), dict)
        )
    }

    if sections.analysis_results and metrics_entries:
        has_content = True
        _draw_section_heading(pdf, "Analysis Results")

        for node_id, entry in metrics_entries.items():
            node_type = entry.get("node_type", node_id)
            section_title = node_type.replace("_", " ").title()

            # Section heading
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(240, 240, 248)
            pdf.cell(
                0, 7, _sanitize_text(f"  {section_title}"),
                fill=True, new_x="LMARGIN", new_y="NEXT",
            )

            # Key-value rows
            metrics = entry["metrics"]
            for key, value in metrics.items():
                label = key.replace("_", " ").title()
                val_str = _format_value(value)
                _draw_kv_row(pdf, label, val_str)

            # Clinical interpretation (if enabled and references exist)
            if sections.clinical_interpretation and node_type in _CLINICAL_REFERENCES:
                refs = _CLINICAL_REFERENCES[node_type]
                for metric_key, (ref_label, ref_range, interpret_fn) in refs.items():
                    metric_val = metrics.get(metric_key)
                    if metric_val is None or not isinstance(metric_val, (int, float)):
                        continue

                    status, explanation = interpret_fn(metric_val)
                    r, g, b = _INTERPRETATION_COLORS.get(status, (0, 0, 0))

                    pdf.ln(1)
                    pdf.set_font("Helvetica", "I", 8)
                    pdf.set_text_color(100, 100, 100)
                    pdf.cell(
                        65, 5, _sanitize_text(f"  Reference Range:"),
                    )
                    pdf.set_font("Helvetica", "", 8)
                    pdf.cell(
                        0, 5, _sanitize_text(ref_range),
                        new_x="LMARGIN", new_y="NEXT",
                    )

                    # Status indicator
                    pdf.set_font("Helvetica", "B", 8)
                    pdf.set_text_color(r, g, b)
                    status_label = status.upper()
                    pdf.cell(65, 5, _sanitize_text(f"  Interpretation:"))
                    pdf.set_font("Helvetica", "", 8)
                    pdf.cell(
                        0, 5,
                        _sanitize_text(f"[{status_label}] {explanation}"),
                        new_x="LMARGIN", new_y="NEXT",
                    )
                    pdf.set_text_color(0, 0, 0)

            pdf.ln(3)

    # === Visualizations ===
    plot_entries = {
        node_id: entry
        for node_id, entry in node_results.items()
        if (
            entry.get("status") == "success"
            and isinstance(entry.get("data"), str)
            and entry["data"].startswith("data:image/png;base64,")
        )
    }

    if sections.visualizations and plot_entries:
        has_content = True
        _draw_section_heading(pdf, "Visualizations")

        for node_id, entry in plot_entries.items():
            node_type = entry.get("node_type", node_id)
            section_title = node_type.replace("_", " ").title()

            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(240, 248, 240)
            pdf.cell(
                0, 7, _sanitize_text(f"  {section_title}"),
                fill=True, new_x="LMARGIN", new_y="NEXT",
            )
            pdf.ln(2)

            # Decode base64 PNG and embed as image
            data_uri = entry["data"]
            if not isinstance(data_uri, str) or "," not in data_uri:
                continue  # skip malformed plot data
            try:
                png_b64 = data_uri.split(",", 1)[1]
                png_bytes = base64.b64decode(png_b64)
            except Exception:
                continue  # skip undecodable plot data
            img_buf = io.BytesIO(png_bytes)

            available_width = pdf.w - pdf.l_margin - pdf.r_margin
            pdf.image(img_buf, x=pdf.l_margin, w=available_width)
            pdf.ln(4)

    # === Audit Trail ===
    if sections.audit_trail and audit_log:
        has_content = True
        _draw_section_heading(pdf, "Parameter Change History")

        # Table header
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(230, 230, 240)
        col_widths = [25, 45, 35, 35, 35]
        headers = ["Time", "Node", "Parameter", "Old Value", "New Value"]
        for w, h in zip(col_widths, headers):
            pdf.cell(w, 6, _sanitize_text(h), border=1, fill=True)
        pdf.ln()

        # Table rows
        pdf.set_font("Helvetica", "", 7)
        for entry in audit_log:
            timestamp = str(entry.get("timestamp", ""))
            node_name = str(entry.get("nodeDisplayName", entry.get("nodeId", "")))
            param_label = str(entry.get("paramLabel", ""))
            old_val = str(entry.get("oldValue", ""))
            new_val = str(entry.get("newValue", ""))
            unit = entry.get("unit", "")
            if unit:
                old_val = f"{old_val} {unit}"
                new_val = f"{new_val} {unit}"

            # Truncate long values for table
            vals = [timestamp[:8], node_name[:20], param_label[:16], old_val[:16], new_val[:16]]
            for w, v in zip(col_widths, vals):
                pdf.cell(w, 5, _sanitize_text(v), border=1)
            pdf.ln()

        pdf.ln(3)
        _draw_hr(pdf)

    # === Clinician Notes ===
    if sections.notes and notes.strip():
        has_content = True
        _draw_section_heading(pdf, "Clinician Notes")
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, _sanitize_text(notes.strip()), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # Empty report guard
    if not has_content:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(140, 140, 140)
        pdf.cell(
            0, 10,
            "No metrics or visualization results found. "
            "Run the pipeline first, then generate the report.",
            new_x="LMARGIN", new_y="NEXT", align="C",
        )

    # Footer
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(160, 160, 160)
    pdf.cell(
        0, 5,
        "Generated by Oscilloom -- for research purposes only. "
        "Clinical decisions require independent professional review.",
        new_x="LMARGIN", new_y="NEXT", align="C",
    )

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/report", summary="Generate a PDF clinical report from pipeline results")
def generate_report(request: ReportRequest) -> Response:
    """
    Generates a PDF clinical report from a prior pipeline execution.

    Call POST /pipeline/execute first to get node_results, then pass those
    results here along with optional report metadata (title, patient_id,
    clinic_name). The endpoint scans node_results for all metrics (output_type
    == "dict") and plot (base64 PNG) outputs and bundles them into a PDF.

    Enhanced fields (optional):
      - session_info: recording metadata for Data Quality section
      - pipeline_config: list of processing steps for Pipeline Configuration
      - audit_log: parameter change history for Audit Trail
      - notes: free-text clinician observations
      - sections: toggle individual report sections on/off

    Returns a binary application/pdf response ready for browser download.

    Raises 503 if fpdf2 is not installed.
    """
    if not _FPDF_AVAILABLE:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail=(
                "PDF generation requires fpdf2. "
                "Install it with: pip install 'fpdf2>=2.7.0'"
            ),
        )

    pdf_bytes = _generate_pdf(
        node_results=request.node_results,
        title=request.title or "Oscilloom EEG Report",
        patient_id=request.patient_id or "",
        clinic_name=request.clinic_name or "",
        session_info=request.session_info,
        pipeline_config=request.pipeline_config,
        audit_log=request.audit_log,
        notes=request.notes or "",
        sections=request.sections,
        included_nodes=request.included_nodes,
    )

    safe_title = (request.title or "report").replace(" ", "_").strip("_") or "report"
    filename = f"{safe_title}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
