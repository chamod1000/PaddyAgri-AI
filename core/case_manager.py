"""
Case Management & Diagnosis History Module (core/case_manager.py)

Implements Case Management System with Repository Persistence Abstraction.
Stores, retrieves, and tracks disease progression across farmer diagnosis cases.
Uses JSON storage abstraction by default.
"""

from abc import ABC, abstractmethod
from datetime import datetime
import json
import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════
class DiagnosisSnapshot(BaseModel):
    """Snapshot of a single diagnosis within a case."""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    diagnosis: str = Field(..., description="Identified disease name")
    confidence: str = Field(default="Medium", description="Diagnostic confidence")
    summary: str = Field(default="", description="Key findings summary")


class FollowUpRecord(BaseModel):
    """Follow-up observation & progression record."""
    follow_up_id: str = Field(..., description="Unique follow-up ID")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    notes: str = Field(default="", description="User notes or findings")
    comparison_summary: str = Field(default="Stable", description="Progression: Improved, Stable, Worsened, Different Disease")


class CaseRecord(BaseModel):
    """Complete Case Record encapsulating historical diagnoses and progression."""
    case_id: str = Field(..., description="Unique case identifier")
    session_id: str = Field(..., description="Session identifier")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    disease: str = Field(default="Paddy Disease", description="Primary diagnosed disease")
    confidence: str = Field(default="Medium", description="Confidence level")
    fertilizer_plan: str = Field(default="", description="NPK fertilizer summary")
    safety_status: str = Field(default="COMPLIANT", description="Pesticide Act safety status")
    quality_score: float = Field(default=1.0, description="Evaluation quality score")
    image_count: int = Field(default=0, description="Total images uploaded")
    snapshots: List[DiagnosisSnapshot] = Field(default_factory=list, description="Historical snapshots")
    follow_ups: List[FollowUpRecord] = Field(default_factory=list, description="Follow-up history")


# ══════════════════════════════════════════════
# REPOSITORY INTERFACE (ABSTRACTION)
# ══════════════════════════════════════════════
class ICaseRepository(ABC):
    """Repository interface for Case Management persistence backends (SQLite / JSON / Cloud DB)."""

    @abstractmethod
    def save_case(self, case: CaseRecord) -> None:
        pass

    @abstractmethod
    def get_case(self, case_id: str) -> Optional[CaseRecord]:
        pass

    @abstractmethod
    def list_cases(self) -> List[CaseRecord]:
        pass

    @abstractmethod
    def search_cases(self, query: str) -> List[CaseRecord]:
        pass


# ══════════════════════════════════════════════
# DEFAULT REPOSITORY IMPLEMENTATION (JSON STORAGE)
# ══════════════════════════════════════════════
class JSONCaseRepository(ICaseRepository):
    """File-backed JSON repository implementation."""

    def __init__(self, storage_path: str = "Data/Cases/case_records.json"):
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        if not os.path.exists(self.storage_path):
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load_data(self) -> Dict[str, dict]:
        if not os.path.exists(self.storage_path):
            return {}
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            # Preserve the corrupt file as a recoverable backup instead of
            # letting the next save_case silently overwrite all case history.
            backup_path = f"{self.storage_path}.corrupt.{datetime.now().strftime('%Y%m%d%H%M%S')}"
            try:
                os.replace(self.storage_path, backup_path)
            except OSError:
                pass
            return {}

    def _save_data(self, data: Dict[str, dict]) -> None:
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_case(self, case: CaseRecord) -> None:
        data = self._load_data()
        data[case.case_id] = case.model_dump()
        self._save_data(data)

    def get_case(self, case_id: str) -> Optional[CaseRecord]:
        data = self._load_data()
        raw = data.get(case_id)
        if raw:
            return CaseRecord(**raw)
        return None

    def list_cases(self) -> List[CaseRecord]:
        data = self._load_data()
        records = [CaseRecord(**raw) for raw in data.values()]
        records.sort(key=lambda c: c.updated_at, reverse=True)
        return records

    def search_cases(self, query: str) -> List[CaseRecord]:
        q = query.lower().strip()
        if not q:
            return self.list_cases()
        all_cases = self.list_cases()
        results = []
        for case in all_cases:
            if (q in case.case_id.lower() or
                q in case.disease.lower() or
                q in case.session_id.lower() or
                q in case.created_at.lower()):
                results.append(case)
        return results


# ══════════════════════════════════════════════
# CASE MANAGER DOMAIN SERVICE
# ══════════════════════════════════════════════
class CaseManager:
    """
    Domain service for managing disease cases and tracking progression.
    Uses dependency injection for repository storage backend.
    """

    def __init__(self, repository: Optional[ICaseRepository] = None):
        self.repository = repository if repository else JSONCaseRepository()

    def record_diagnosis(self, response_obj: Any) -> CaseRecord:
        """Extracts structured outputs and persists or updates the CaseRecord."""
        ctx = getattr(response_obj, "processing_context", None)
        mem = getattr(ctx, "conversation_memory", None) if ctx else None
        case_mem = getattr(mem, "case_memory", None) if mem else None
        diag = getattr(response_obj, "diagnostic_info", None)
        fert = getattr(response_obj, "fertilizer_info", None)
        refl = getattr(response_obj, "reflection_result", None)
        eval_res = getattr(response_obj, "evaluation_result", None)
        meta = getattr(ctx, "metadata", {}) if ctx else {}

        sess_id = meta.get("session_id", "sess_default")
        case_id = getattr(case_mem, "case_id", f"case_{sess_id[:8]}") if case_mem else f"case_{sess_id[:8]}"

        disease_name = getattr(diag, "suspected_disease", "General Pathology Inspection") if diag else "General Paddy Query"
        confidence = getattr(diag, "confidence_level", "High") if diag else "High"
        
        fert_str = ""
        if fert:
            fert_str = f"Urea {getattr(fert, 'urea_dosage_per_acre_kg', 0)}kg, TSP {getattr(fert, 'tsp_dosage_per_acre_kg', 0)}kg, MOP {getattr(fert, 'mop_dosage_per_acre_kg', 0)}kg"

        safety_str = "COMPLIANT" if (refl and getattr(refl, "all_checks_passed", True)) else "VERIFICATION_WARNING"
        q_score = getattr(getattr(eval_res, "overall_eval", None), "composite_quality_score", 1.0) if eval_res else 1.0

        existing = self.repository.get_case(case_id)
        now_str = datetime.now().isoformat()

        snapshot = DiagnosisSnapshot(
            timestamp=now_str,
            diagnosis=disease_name,
            confidence=confidence,
            summary=f"Diagnosed {disease_name} with {confidence} confidence."
        )

        if existing:
            existing.updated_at = now_str
            existing.disease = disease_name
            existing.confidence = confidence
            if fert_str: existing.fertilizer_plan = fert_str
            existing.snapshots.append(snapshot)
            
            # Determine deterministic progression summary without calling LLM
            prev_diag = existing.snapshots[-2].diagnosis if len(existing.snapshots) >= 2 else disease_name
            if prev_diag.lower() == disease_name.lower():
                comp = "Stable (Same Pathology)"
            else:
                comp = f"Different Disease Suspected ({prev_diag} -> {disease_name})"

            follow_up = FollowUpRecord(
                follow_up_id=f"fol_{len(existing.follow_ups)+1}",
                timestamp=now_str,
                notes=f"Re-inspected paddy leaf. Current: {disease_name}.",
                comparison_summary=comp
            )
            existing.follow_ups.append(follow_up)
            self.repository.save_case(existing)
            return existing
        else:
            new_case = CaseRecord(
                case_id=case_id,
                session_id=sess_id,
                created_at=now_str,
                updated_at=now_str,
                disease=disease_name,
                confidence=confidence,
                fertilizer_plan=fert_str,
                safety_status=safety_str,
                quality_score=q_score,
                image_count=getattr(case_mem, "uploaded_images_count", 1) if case_mem else 1,
                snapshots=[snapshot],
                follow_ups=[]
            )
            self.repository.save_case(new_case)
            return new_case

    def get_history(self) -> List[CaseRecord]:
        return self.repository.list_cases()

    def search_history(self, query: str) -> List[CaseRecord]:
        return self.repository.search_cases(query)
