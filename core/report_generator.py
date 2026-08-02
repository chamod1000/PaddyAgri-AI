"""
Professional PDF Report Generation Module (core/report_generator.py)

Generates clean, enterprise-grade Crop Health & Pathology Diagnostic Reports as PDF documents.
Consumes structured AgentResponse, ProcessingContext, RequestTrace, and EvaluationResult objects.
Uses fpdf2 for zero-dependency PDF rendering.
"""

from datetime import datetime
from io import BytesIO
from typing import Any, Optional
from fpdf import FPDF


class CropHealthPDF(FPDF):
    """Custom FPDF document layout for Crop Health Diagnostic Reports."""

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(26, 77, 46)  # Deep emerald green
        self.cell(0, 8, "SRI LANKAN PADDY AGRI-AI PLATFORM", new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, "Official Multi-Agent Crop Health & Pathology Diagnostic Report", new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()} | Generated automatically by PaddyAgri-AI Multi-Agent System", align="C")

    def chapter_title(self, title: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(26, 77, 46)
        self.set_fill_color(240, 247, 240)
        self.cell(0, 7, f"  {title}", new_x="LMARGIN", new_y="NEXT", align="L", fill=True)
        self.ln(2)

    def key_value_row(self, label: str, value: str):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(50, 50, 50)
        self.cell(50, 6, label, new_x="RIGHT", new_y="TOP")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")


class ReportGenerator:
    """
    Enterprise PDF Report Generator.
    Consumes structured AgentResponse and ProcessingContext payloads to render PDF reports.
    """

    @staticmethod
    def generate_pdf(response_obj: Any) -> bytes:
        pdf = CropHealthPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Extract underlying structured context objects
        ctx = getattr(response_obj, "processing_context", None)
        trace = getattr(response_obj, "request_trace", None)
        eval_res = getattr(response_obj, "evaluation_result", None)
        mem = getattr(ctx, "conversation_memory", None) if ctx else None
        case_mem = getattr(mem, "case_memory", None) if mem else None
        meta = getattr(ctx, "metadata", {}) if ctx else {}
        vis_analysis = getattr(response_obj, "vision_info", None)
        diag = getattr(response_obj, "diagnostic_info", None)
        fert = getattr(response_obj, "fertilizer_info", None)
        refl = getattr(response_obj, "reflection_result", None)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        req_id = getattr(trace, "request_id", meta.get("request_id", "req_unknown"))
        sess_id = getattr(trace, "session_id", meta.get("session_id", "sess_unknown"))
        case_id = getattr(case_mem, "case_id", f"case_{sess_id[:8]}") if case_mem else "case_default"

        # 1. Report Metadata & Header Block
        pdf.chapter_title("1. DOCUMENT CONTROL & REPORT METADATA")
        pdf.key_value_row("Report Generated:", now_str)
        pdf.key_value_row("Case Identifier:", str(case_id))
        pdf.key_value_row("Session Identifier:", str(sess_id))
        pdf.key_value_row("Request Tracking ID:", str(req_id))
        pdf.ln(3)

        # 2. Farmer Query & Input Modality
        pdf.chapter_title("2. FARMER REQUEST & INPUT MODALITY")
        query_text = getattr(response_obj, "query", "Paddy diagnosis request")
        has_img = meta.get("has_image", False) or vis_analysis is not None
        pdf.key_value_row("User Query Text:", query_text)
        pdf.key_value_row("Photo Attachment:", "Yes (Paddy Leaf Image Uploaded)" if has_img else "No (Text Query Only)")
        pdf.ln(3)

        # 3. Vision Pathology Analysis (If Image present)
        if vis_analysis:
            pdf.chapter_title("3. VISUAL PATHOLOGY OBSERVATIONS (VISION LAYER)")
            symptoms_list = getattr(vis_analysis, "visible_symptoms", [])
            symptom_str = ", ".join(symptoms_list) if symptoms_list else "Not detected"
            pdf.key_value_row("Image Quality:", getattr(vis_analysis, "image_quality", "Good"))
            pdf.key_value_row("Leaf Color:", getattr(vis_analysis, "leaf_color", "Green with lesions"))
            pdf.key_value_row("Visible Symptoms:", symptom_str)
            pdf.key_value_row("Spot Characteristics:", getattr(vis_analysis, "spot_characteristics", "Lesions detected"))
            pdf.key_value_row("Pattern Distribution:", getattr(vis_analysis, "pattern_distribution", "Foliar distribution"))
            pdf.key_value_row("Extraction Confidence:", getattr(vis_analysis, "confidence_estimate", "HIGH"))
            pdf.ln(3)

        # 4. Disease Diagnosis (DiagnosticAgent Output)
        if diag:
            pdf.chapter_title("4. PATHOLOGY DIAGNOSIS (DIAGNOSTIC AGENT)")
            pdf.key_value_row("Suspected Disease:", getattr(diag, "suspected_disease", "Unknown Disease"))
            pdf.key_value_row("Confidence Level:", getattr(diag, "confidence_level", "Medium"))
            treatments = getattr(diag, "treatment_recommended", [])
            treatment_str = "; ".join(treatments) if treatments else "Follow DOA guidelines"
            pdf.key_value_row("Recommended Treatments:", treatment_str)
            pdf.ln(3)

        # 5. Fertilizer Recommendation (FertilizerAgent Output)
        if fert:
            pdf.chapter_title("5. FERTILIZER & NPK DOSAGE RECOMMENDATION")
            pdf.key_value_row("Cultivation Season:", getattr(fert, "season", "Maha / Yala"))
            pdf.key_value_row("Urea (N) Rate:", f"{getattr(fert, 'urea_dosage_per_acre_kg', 0.0)} kg / acre")
            pdf.key_value_row("TSP (P) Rate:", f"{getattr(fert, 'tsp_dosage_per_acre_kg', 0.0)} kg / acre")
            pdf.key_value_row("MOP (K) Rate:", f"{getattr(fert, 'mop_dosage_per_acre_kg', 0.0)} kg / acre")
            schedule = getattr(fert, "application_schedule", [])
            schedule_str = "; ".join(schedule) if schedule else "Standard split application"
            pdf.key_value_row("Application Schedule:", schedule_str)
            pdf.ln(3)

        # 6. Safety & Regulatory Compliance (ReflectionAgent Output)
        if refl:
            pdf.chapter_title("6. SAFETY & REGULATORY REVIEW (REFLECTION AGENT)")
            all_passed = getattr(refl, "all_checks_passed", True)
            pdf.key_value_row("Pesticide Act No. 33:", "COMPLIANT (DOA Approved Chemicals Only)" if all_passed else "SAFETY VERIFICATION WARNING")
            warnings = getattr(refl, "warnings", [])
            warn_str = "; ".join(warnings) if warnings else "None (All safety checks passed cleanly)"
            pdf.key_value_row("Safety Advisory Notes:", warn_str)
            citations = getattr(refl, "regulatory_citations", [])
            cit_str = "; ".join(citations) if citations else "Sri Lanka Dept. of Agriculture (DOA)"
            pdf.key_value_row("Regulatory Citations:", cit_str)
            pdf.ln(3)

        # 7. Historical Case Intelligence Summary
        if case_mem:
            pdf.chapter_title("7. CROP CASE HISTORY & PROGRESSION METRICS")
            prev_diag = getattr(case_mem, "previous_diagnoses", [])
            pdf.key_value_row("Historical Diagnoses:", ", ".join(prev_diag) if prev_diag else "First diagnostic entry for case")
            pdf.key_value_row("Cumulative Photos Inspected:", str(getattr(case_mem, "uploaded_images_count", 1)))
            pdf.ln(3)

        # 8. System Performance & Quality Benchmarks
        pdf.chapter_title("8. SYSTEM PERFORMANCE & EVALUATION BENCHMARKS")
        perf = getattr(trace, "performance", None) if trace else None
        total_latency = getattr(perf, "total_latency_ms", 0.0) if perf else 0.0
        pdf.key_value_row("End-to-End Latency:", f"{total_latency:.2f} ms")
        
        overall_eval = getattr(eval_res, "overall_eval", None) if eval_res else None
        if overall_eval:
            pdf.key_value_row("Quality Tier:", getattr(overall_eval, "quality_tier", "EXCELLENT"))
            pdf.key_value_row("Composite Quality Score:", f"{getattr(overall_eval, 'composite_quality_score', 1.0):.2f} / 1.0")
        pdf.ln(3)

        # 9. Environmental Weather & Seasonal Advisories
        w_ctx = getattr(response_obj, "weather_info", None)
        s_adv = getattr(response_obj, "seasonal_advisory", None)
        if w_ctx:
            pdf.chapter_title("9. ENVIRONMENTAL WEATHER & SEASONAL ADVISORIES")
            pdf.key_value_row("Target Location:", getattr(w_ctx, "location", "North Central Province"))
            pdf.key_value_row("Season / Temp / Humidity:", f"{getattr(w_ctx, 'season', 'Yala')} | {getattr(w_ctx, 'temperature_c', 31.0)} C | {getattr(w_ctx, 'humidity_pct', 80.0)}% RH")
            pdf.key_value_row("Rainfall Probability:", f"{getattr(w_ctx, 'rainfall_probability_pct', 50.0)}% (24-hour forecast)")
            if s_adv:
                pdf.key_value_row("Fungal Disease Risk:", getattr(s_adv, "fungal_risk_alert", "Normal"))
                pdf.key_value_row("Fertilizer Timing Advice:", getattr(s_adv, "fertilizer_timing_advice", "Standard"))
                notes = getattr(s_adv, "advisory_notes", [])
                pdf.key_value_row("Seasonal Rule Guidance:", "; ".join(notes) if notes else "Follow standard practices")

        # 10. Explainable AI (XAI) Rationale
        expl = getattr(response_obj, "explanation", None)
        if expl:
            pdf.ln(3)
            pdf.chapter_title("10. EXPLAINABLE AI (XAI) ANALYSIS & RATIONALE")
            pdf.key_value_row("Primary Disease Target:", getattr(expl, "disease_name", "Paddy Pathology"))
            pdf.key_value_row("Diagnostic Rationale:", getattr(expl, "recommendation_rationale", "Evidence mapped cleanly."))
            pdf.key_value_row("Explanation Summary:", getattr(expl, "explanation_summary", "Diagnostic evidence aligned."))

        buffer = BytesIO()
        pdf.output(buffer)
        return buffer.getvalue()
