

from __future__ import annotations

import datetime
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any, Tuple

from ml.report.metadata import ConfidenceLevel, ReportMetadata
from backend.app.config.settings import settings

logger = logging.getLogger("brain_tumor_api")

CLINICAL_DESCRIPTIONS = {
    "glioma": (
        "Gliomas are primary brain tumors originating from glial cells. They can range from low-grade "
        "to highly aggressive high-grade glioblastomas. Common radiographic features include irregular contrast "
        "enhancement, central necrosis, surrounding edema, and mass effect. T2/FLAIR hyperintensity indicates "
        "infiltrative tumor cells and vasogenic edema."
    ),
    "meningioma": (
        "Meningiomas are typically extra-axial, slow-growing tumors arising from the arachnoid cap cells of the meninges. "
        "Radiographic hallmarks include a well-circumscribed dural-based mass, intense uniform contrast enhancement, and "
        "frequently a 'dural tail' sign. They may cause hyperostosis of adjacent bone and compression of adjacent brain parenchyma."
    ),
    "pituitary": (
        "Pituitary tumors (commonly adenomas) are sella-based lesions originating from the anterior pituitary gland. "
        "They are categorized as microadenomas (<10mm) or macroadenomas (>=10mm). MRI findings typically show sellar enlargement, "
        "expansion into the suprasellar cistern, compression of the optic chiasm ('snowman' or 'figure-of-8' shape), and variable "
        "contrast enhancement compared to normal pituitary tissue."
    ),
    "no tumor": (
        "No radiographic evidence of intracranial mass, abnormal contrast enhancement, or diagnostic features of glioma, meningioma, "
        "or pituitary tumor was identified in the analyzed scan. Ventricles, sulci, and extra-axial spaces appear within normal anatomical limits."
    ),
    "notumor": (
        "No radiographic evidence of intracranial mass, abnormal contrast enhancement, or diagnostic features of glioma, meningioma, "
        "or pituitary tumor was identified in the analyzed scan. Ventricles, sulci, and extra-axial spaces appear within normal anatomical limits."
    )
}

CLINICAL_RECOMMENDATIONS = {
    "glioma": [
        "Urgent neurosurgical consultation for biopsy/resection planning.",
        "Obtain advanced MRI sequencing including MR Spectroscopy and Perfusion imaging.",
        "Consider initiation of anti-edema therapy (e.g., Dexamethasone) if significant mass effect is present.",
        "Neurology consultation for seizure prophylaxis if clinically indicated."
    ],
    "meningioma": [
        "Neurosurgical consultation to evaluate for potential resection vs. observation.",
        "Contrast-enhanced follow-up MRI in 3-6 months to assess growth kinetics.",
        "Symptomatic management of mass effect or headache if present.",
        "Evaluation of adjacent venous sinuses for potential invasion or compromise."
    ],
    "pituitary": [
        "Endocrinology consultation for complete pituitary hormone panel evaluation.",
        "Neurosurgical consultation for transsphenoidal resection if mass effect or optic chiasm compression is noted.",
        "Formal visual field testing (perimetry) to evaluate for bitemporal hemianopsia.",
        "High-resolution dedicated coronal and sagittal MRI of the sella turcica."
    ],
    "no tumor": [
        "Routine clinical follow-up as indicated by primary symptoms.",
        "No neurosurgical or oncological intervention is warranted based on this scan.",
        "Correlate clinically with patient's initial presenting symptoms (headache, dizziness, etc.)."
    ],
    "notumor": [
        "Routine clinical follow-up as indicated by primary symptoms.",
        "No neurosurgical or oncological intervention is warranted based on this scan.",
        "Correlate clinically with patient's initial presenting symptoms (headache, dizziness, etc.)."
    ]
}

class ReportService:

    def generate_report(
        self,
        prediction: str,
        confidence: float,
        probabilities: Dict[str, float],
        original_image_path: Path,
        heatmap_path: Path,
        overlay_path: Path,
        inference_time_ms: float,
        device: str,
        model_version: str
    ) -> Dict[str, str]:

        meta = ReportMetadata.generate_unique()
        report_id = meta.report_id
        
        results_root = Path("ml/artifacts/results").resolve()
        report_dir = results_root / report_id
        report_dir.mkdir(parents=True, exist_ok=True)

        dest_original = report_dir / "original.jpg"
        dest_heatmap = report_dir / "heatmap.png"
        dest_overlay = report_dir / "overlay.png"

        shutil.copy2(original_image_path, dest_original)
        shutil.copy2(heatmap_path, dest_heatmap)
        shutil.copy2(overlay_path, dest_overlay)

        qualitative_confidence = ConfidenceLevel.get_level(confidence)

        display_probs = {
            k.title() if k.lower() != "notumor" else "No Tumor": float(v)
            for k, v in probabilities.items()
        }

        pred_key = prediction.lower().strip()
        description = CLINICAL_DESCRIPTIONS.get(pred_key, CLINICAL_DESCRIPTIONS["no tumor"])
        recs = CLINICAL_RECOMMENDATIONS.get(pred_key, CLINICAL_RECOMMENDATIONS["no tumor"])

        json_data = {
            "report_id": report_id,
            "timestamp": meta.generation_date.isoformat(),
            "model_metadata": {
                "architecture": "EfficientNet-B0",
                "version": model_version,
                "inference_time_ms": round(inference_time_ms, 2)
            },
            "findings": {
                "prediction": prediction.title() if prediction.lower() != "notumor" else "No Tumor",
                "confidence_score": round(confidence, 2),
                "confidence_level": qualitative_confidence,
                "posterior_probabilities": display_probs
            },
            "clinical_notes": {
                "pathological_description": description,
                "actionable_recommendations": recs,
                "evaluations_checklist": [
                    "Mass Effect / Herniation",
                    "Surrounding Vasogenic Edema",
                    "Midline Shift",
                    "Ventricular Compression",
                    "Dural Attachment",
                    "Sellar / Suprasellar Extension",
                    "Contrast Enhancement Pattern"
                ]
            },
            "legal_disclaimer": (
                "This report was automatically generated by an Artificial Intelligence system for research, "
                "educational, and clinical decision-support purposes. The prediction represents the output of "
                "a deep learning model based on the provided MRI image and should not be interpreted as a "
                "definitive medical diagnosis."
            )
        }
        
        json_path = report_dir / "analysis_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        txt_path = report_dir / "analysis_summary.txt"
        self._write_txt_summary(txt_path, json_data, meta)

        html_path = report_dir / "analysis_report.html"
        self._write_html_report(html_path, json_data, meta, dest_original, dest_heatmap, dest_overlay)

        pdf_path = report_dir / "analysis_report.pdf"
        self._write_pdf_report(pdf_path, json_data, meta, dest_overlay)

        reports_dir = Path(settings.MODEL_CHECKPOINT_DIR).parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(json_path, reports_dir / "analysis_report.json")
        shutil.copy2(txt_path, reports_dir / "analysis_summary.txt")
        shutil.copy2(html_path, reports_dir / "analysis_report.html")
        shutil.copy2(pdf_path, reports_dir / "analysis_report.pdf")

        logger.info("Generated report suite for ID %s successfully", report_id)
        
        return {
            "report_id": report_id,
            "original_image": str(dest_original),
            "heatmap_image": str(dest_heatmap),
            "overlay_image": str(dest_overlay),
            "report_json": str(json_path),
            "report_txt": str(txt_path),
            "report_html": str(html_path),
            "report_pdf": str(pdf_path)
        }

    def _write_txt_summary(self, path: Path, data: Dict[str, Any], meta: ReportMetadata) -> None:
        
        findings = data["findings"]
        notes = data["clinical_notes"]
        model = data["model_metadata"]

        summary = f"""======================================================================
                     NEUROVISION AI LAB - CLINICAL REPORT
======================================================================
Report ID          : {data['report_id']}
Generation Date    : {meta.generation_date.strftime("%Y-%m-%d %H:%M:%S")}
Model Version      : {model['version']}
----------------------------------------------------------------------
ANALYSIS RESULT:
Predicted Class    : {findings['prediction']}
Confidence Score   : {findings['confidence_score']:.2f}% ({findings['confidence_level']})

CLINICAL DESCRIPTION:
{notes['pathological_description']}

RECOMMENDED ACTION ITEMS:
"""
        for i, rec in enumerate(notes['actionable_recommendations'], 1):
            summary += f"  {i}. [ ] {rec}\n"

        summary += """
CLINICAL FINDINGS CHECKLIST:
"""
        for chk in notes['evaluations_checklist']:
            summary += f"  [ ] {chk}\n"

        summary += """----------------------------------------------------------------------
PROBABILITY DISTRIBUTION:
"""
        for cls, prob in findings['posterior_probabilities'].items():
            summary += f"  - {cls:<15}: {prob:6.2f}%\n"

        summary += f"""----------------------------------------------------------------------
MODEL ENVIRONMENT:
Architecture       : EfficientNet-B0
Inference Latency  : {model['inference_time_ms']:.1f} ms
----------------------------------------------------------------------
AUTHORIZED SIGN-OFF:

Reviewing Physician: ___________________________
Signature / Date   : ___________________________

MEDICAL DISCLAIMER:
{data['legal_disclaimer']}
======================================================================
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(summary)

    def _write_html_report(
        self,
        path: Path,
        data: Dict[str, Any],
        meta: ReportMetadata,
        orig: Path,
        heatmap: Path,
        overlay: Path
    ) -> None:
        
        orig_rel = orig.name
        heatmap_rel = heatmap.name
        overlay_rel = overlay.name

        findings = data["findings"]
        notes = data["clinical_notes"]
        model = data["model_metadata"]

        pred = findings["prediction"].lower()
        if "glioma" in pred:
            accent_color = "#ef4444"
            accent_bg = "rgba(239,68,68,0.1)"
        elif "meningioma" in pred:
            accent_color = "#f59e0b"
            accent_bg = "rgba(245,158,11,0.1)"
        elif "pituitary" in pred:
            accent_color = "#8b5cf6"
            accent_bg = "rgba(139,92,246,0.1)"
        else:
            accent_color = "#10b981"
            accent_bg = "rgba(16,185,129,0.1)"

        prob_rows = ""
        for cls, prob in findings['posterior_probabilities'].items():
            is_pred = cls == findings['prediction']
            weight = "font-weight: 700; color: #00d4e8;" if is_pred else ""
            marker = " (Predicted)" if is_pred else ""
            prob_rows += f"""
            <tr style="{weight}">
                <td style="padding: 12px; border-bottom: 1px solid #1a2d42; font-family: monospace;">{cls}</td>
                <td style="padding: 12px; border-bottom: 1px solid #1a2d42; width: 60%;">
                    <div style="background: #1a2d42; border-radius: 4px; height: 8px; width: 100%; overflow: hidden;">
                        <div style="background: {accent_color if is_pred else '#00d4e8'}; height: 100%; width: {prob}%;"></div>
                    </div>
                </td>
                <td style="padding: 12px; border-bottom: 1px solid #1a2d42; text-align: right; font-family: monospace;">{prob:.2f}%{marker}</td>
            </tr>"""

        rec_list_items = "".join([f"<li style='margin-bottom: 8px;'>{rec}</li>" for rec in notes['actionable_recommendations']])
        chk_items = "".join([
            f"<div style='display: flex; align-items: center; gap: 10px; margin-bottom: 10px; font-family: monospace; font-size: 13px;'>"
            f"<div style='width: 14px; height: 14px; border: 1px solid #1a2d42; border-radius: 3px; background: #080d12;'></div>"
            f"<span>{chk}</span>"
            f"</div>"
            for chk in notes['evaluations_checklist']
        ])

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Brain MRI Clinical Report - {data['report_id']}</title>
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: #080d12;
            color: #e2e8f0;
            margin: 0;
            padding: 30px 20px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: #0e1621;
            border-radius: 12px;
            border: 1px solid #1a2d42;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            padding: 40px;
        }}
        .header {{
            border-bottom: 1px solid #1a2d42;
            padding-bottom: 24px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header-logo {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .logo-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #00d4e8;
            box-shadow: 0 0 10px #00d4e8;
        }}
        .header h1 {{
            color: #00d4e8;
            margin: 0;
            font-size: 24px;
            font-family: monospace;
            letter-spacing: -0.5px;
        }}
        .header p {{
            margin: 5px 0 0 0;
            color: #7a9bb5;
            font-size: 13px;
        }}
        .meta-box {{
            text-align: right;
            font-size: 13px;
            color: #7a9bb5;
            font-family: monospace;
            line-height: 1.5;
        }}
        .card {{
            background: #111d2b;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 30px;
            border-left: 4px solid {accent_color};
            border-top: 1px solid #1a2d42;
            border-right: 1px solid #1a2d42;
            border-bottom: 1px solid #1a2d42;
        }}
        .card h2 {{
            margin-top: 0;
            color: #00d4e8;
            font-size: 14px;
            font-family: monospace;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 12px;
        }}
        .grid-layout {{
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        .section-title {{
            color: #00d4e8;
            font-size: 13px;
            font-family: monospace;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            border-bottom: 1px solid #1a2d42;
            padding-bottom: 8px;
            margin-bottom: 16px;
        }}
        .grid-images {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 25px 0;
        }}
        .grid-images div {{
            text-align: center;
        }}
        .grid-images img {{
            width: 100%;
            border-radius: 8px;
            border: 1px solid #1a2d42;
            background-color: #000;
            transition: transform 0.2s;
        }}
        .grid-images img:hover {{
            transform: scale(1.02);
            border-color: #00d4e8;
        }}
        .table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        .table th {{
            background-color: #111d2b;
            color: #7a9bb5;
            font-family: monospace;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #1a2d42;
        }}
        .sign-off-section {{
            margin-top: 40px;
            border-top: 1px solid #1a2d42;
            padding-top: 30px;
            display: flex;
            justify-content: space-between;
        }}
        .signature-field {{
            width: 45%;
        }}
        .signature-line {{
            border-bottom: 1px dashed #4a6580;
            height: 50px;
            margin-bottom: 10px;
        }}
        .disclaimer {{
            font-size: 11px;
            color: #4a6580;
            border-top: 1px solid #1a2d42;
            padding-top: 20px;
            margin-top: 40px;
            line-height: 1.6;
        }}
        @media print {{
            body {{
                background-color: #ffffff;
                color: #000000;
                padding: 0;
            }}
            .container {{
                border: none;
                box-shadow: none;
                max-width: 100%;
                padding: 0;
                background: none;
            }}
            .card {{
                background: #f8fafc;
                border-color: #cbd5e1;
                color: #000;
            }}
            .grid-images img {{
                border-color: #cbd5e1;
            }}
            .table th {{
                background-color: #f1f5f9;
                color: #334155;
                border-color: #cbd5e1;
            }}
            .table td {{
                border-color: #cbd5e1;
            }}
            .header h1 {{
                color: #1e3a8a;
            }}
            .header-logo .logo-dot {{
                background: #1e3a8a;
            }}
            .signature-line {{
                border-color: #475569;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-logo">
                <div class="logo-dot"></div>
                <div>
                    <h1>BrainCircuit AI</h1>
                    <p>Clinical Brain MRI Analysis Report</p>
                </div>
            </div>
            <div class="meta-box">
                <strong>Report ID:</strong> {data['report_id']}<br>
                <strong>Date:</strong> {meta.generation_date.strftime("%Y-%m-%d %H:%M:%S")}
            </div>
        </div>

        <div class="card">
            <h2>Outcome Summary</h2>
            <div style="font-size: 26px; font-weight: bold; color: {accent_color}; margin: 8px 0; font-family: monospace;">
                {findings['prediction']}
            </div>
            <div style="font-size: 15px; color: #e2e8f0;">
                Model Confidence: <strong style="color: #00d4e8;">{findings['confidence_score']:.2f}%</strong> &mdash; <em>{findings['confidence_level']}</em>
            </div>
        </div>

        <div class="grid-layout">
            <div>
                <div class="section-title">Clinical Findings & Pathological Description</div>
                <p style="font-size: 14px; line-height: 1.7; color: #e2e8f0; text-align: justify; margin: 0 0 20px 0;">
                    {notes['pathological_description']}
                </p>

                <div class="section-title">Actionable Clinical Recommendations</div>
                <ul style="font-size: 14px; line-height: 1.7; color: #e2e8f0; padding-left: 20px; margin: 0;">
                    {rec_list_items}
                </ul>
            </div>

            <div>
                <div class="section-title">Clinical Evaluation Checklist</div>
                <p style="font-size: 11px; color: #7a9bb5; margin-bottom: 15px; font-family: monospace;">For Reviewing Radiologist's mark-up:</p>
                {chk_items}
            </div>
        </div>

        <div>
            <div class="section-title">Posterior Probability Distribution</div>
            <table class="table">
                <thead>
                    <tr>
                        <th style="width: 30%;">Classification Category</th>
                        <th style="width: 50%;">Confidence Meter</th>
                        <th style="width: 20%; text-align: right;">Probability Value</th>
                    </tr>
                </thead>
                <tbody>
                    {prob_rows}
                </tbody>
            </table>
        </div>

        <div style="margin-top: 40px;">
            <div class="section-title">Explainable AI Attribution Visualizations</div>
            <p style="font-size: 13px; color: #7a9bb5; margin-bottom: 15px;">
                The highlighted gradient overlay outlines the pixel zones that contributed most significantly to the model decision.
            </p>
            <div class="grid-images">
                <div>
                    <img src="{orig_rel}" alt="Original MRI">
                    <p style="font-size: 11px; color: #7a9bb5; margin-top: 8px; font-family: monospace;">Original MRI</p>
                </div>
                <div>
                    <img src="{heatmap_rel}" alt="CAM Heatmap">
                    <p style="font-size: 11px; color: #7a9bb5; margin-top: 8px; font-family: monospace;">CAM Heatmap</p>
                </div>
                <div>
                    <img src="{overlay_rel}" alt="CAM Overlay">
                    <p style="font-size: 11px; color: #7a9bb5; margin-top: 8px; font-family: monospace;">CAM Overlay</p>
                </div>
            </div>
        </div>

        <div style="margin-top: 30px; font-size: 13px; color: #7a9bb5; display: flex; gap: 40px; font-family: monospace;">
            <div><strong>Model Architecture:</strong> EfficientNet-B0 (v{model['version']})</div>
            <div><strong>Inference Latency:</strong> {model['inference_time_ms']:.1f} ms</div>
        </div>

        <div class="sign-off-section">
            <div class="signature-field">
                <div class="signature-line"></div>
                <div style="font-size: 12px; color: #e2e8f0; font-weight: bold; font-family: monospace;">Reviewing Radiologist</div>
                <div style="font-size: 11px; color: #7a9bb5;">Printed Name / Title</div>
            </div>
            <div class="signature-field">
                <div class="signature-line"></div>
                <div style="font-size: 12px; color: #e2e8f0; font-weight: bold; font-family: monospace;">Signature & Date</div>
                <div style="font-size: 11px; color: #7a9bb5;">Clinical Sign-off Certification</div>
            </div>
        </div>

        <div class="disclaimer">
            <strong>Medical Disclaimer:</strong> {data['legal_disclaimer']}
        </div>
    </div>
</body>
</html>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _write_pdf_report(
        self,
        path: Path,
        data: Dict[str, Any],
        meta: ReportMetadata,
        overlay: Path
    ) -> None:
        
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        findings = data["findings"]
        notes = data["clinical_notes"]
        model = data["model_metadata"]

        doc = SimpleDocTemplate(
            str(path),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        story = []

        navy = colors.HexColor("#1e3a8a")
        gray = colors.HexColor("#475569")
        light_gray = colors.HexColor("#f1f5f9")

        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor=navy,
            spaceAfter=10
        )
        meta_style = ParagraphStyle(
            "MetaText",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=gray,
            alignment=2  
        )
        h2_style = ParagraphStyle(
            "SectionHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=navy,
            spaceBefore=8,
            spaceAfter=5
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=colors.black,
            spaceAfter=4,
            leading=12
        )
        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7,
            textColor=gray,
            leading=9
        )

        header_data = [
            [Paragraph("NeuroVision AI Lab", title_style),
             Paragraph(f"<b>Report ID:</b> {data['report_id']}<br/><b>Date:</b> {meta.generation_date.strftime('%Y-%m-%d %H:%M')}", meta_style)]
        ]
        header_table = Table(header_data, colWidths=[270, 270])
        header_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, -1), 1.5, navy),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 8))

        result_text = f"""
        <b>Predicted Category:</b> {findings['prediction']}<br/>
        <b>AI Confidence:</b> {findings['confidence_score']:.2f}% ({findings['confidence_level']})
        """
        result_table = Table([[Paragraph(result_text, ParagraphStyle("ResultText", parent=body_style, fontSize=11, leading=15))]], colWidths=[540])
        result_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), light_gray),
            ("BOX", (0, 0), (-1, -1), 1.2, navy),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ]))
        story.append(result_table)
        story.append(Spacer(1, 8))

        col_desc = f"""<b>Pathological Description:</b><br/>{notes['pathological_description']}<br/><br/>
        <b>Clinical Recommendations:</b><br/>"""
        for r in notes['actionable_recommendations']:
            col_desc += f"&bull; {r}<br/>"

        col_chk = "<b>Clinical Checklist (For Radiologist):</b><br/><br/>"
        for chk in notes['evaluations_checklist']:
            col_chk += f"[  ] {chk}<br/><br/>"

        split_table = Table([
            [Paragraph(col_desc, body_style), Paragraph(col_chk, body_style)]
        ], colWidths=[320, 200])
        split_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(split_table)
        story.append(Spacer(1, 8))

        story.append(Paragraph("Category Probabilities", h2_style))
        table_data = [["Category", "Confidence Score"]]
        for cls, prob in findings['posterior_probabilities'].items():
            table_data.append([cls, f"{prob:.2f}%"])
        
        prob_table = Table(table_data, colWidths=[270, 270])
        prob_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), navy),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 8))

        img_flowables = [
            Paragraph("Grad-CAM Overlay Visualization", h2_style),
            Paragraph("The highlighted region displays areas of feature density contributing to prediction:", body_style)
        ]
        try:
            rl_img = RLImage(str(overlay), width=160, height=160)
            img_flowables.append(rl_img)
        except Exception as exc:
            img_flowables.append(Paragraph(f"[Error rendering overlay image: {exc}]", body_style))

        specs_txt = f"""
        <b>Model:</b> EfficientNet-B0 (v{model['version']}) &nbsp;&nbsp;|&nbsp;&nbsp; <b>Latency:</b> {model['inference_time_ms']:.1f} ms
        """
        img_flowables.append(Spacer(1, 4))
        img_flowables.append(Paragraph(specs_txt, body_style))

        sign_data = [
            ["_____________________________________", "_____________________________________"],
            ["Reviewing Radiologist (Printed Name)", "Authorized Signature / Date"]
        ]
        sign_table = Table(sign_data, colWidths=[270, 270])
        sign_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (-1, -1), gray),
        ]))
        img_flowables.append(Spacer(1, 12))
        img_flowables.append(sign_table)

        story.append(KeepTogether(img_flowables))
        story.append(Spacer(1, 8))

        disclaimer_text = f"""
        <b>Medical Disclaimer:</b><br/>
        {data['legal_disclaimer']}
        """
        disclaimer_table = Table([[Paragraph(disclaimer_text, disclaimer_style)]], colWidths=[540])
        disclaimer_table.setStyle(TableStyle([
            ("LINEABOVE", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(disclaimer_table)

        doc.build(story)
