"""
ReflectionAgent Implementation
Agentic Pattern 4: Reflection & Self-Critique Pattern for safety & regulatory verification.
"""

import uuid
from typing import List, Optional

from core.agent_messages import (
    AgentMessage, DiagnosticResult, FertilizerRecommendation,
    SafetyVerdict, ReflectionResult
)
from core.agents.base_agent import BaseAgent


class ReflectionAgent(BaseAgent):
    """
    Safety & Quality Verifier that double-checks pesticide/fertilizer recommendations
    against Sri Lankan Department of Agriculture (DoA) environmental safety guidelines
    and biosecurity regulations before final output synthesis.
    """

    def __init__(self):
        super().__init__(name="ReflectionAgent", model=None)  # No LLM needed for deterministic checks

        # Sri Lankan DoA maximum dosage limits (kg per acre)
        self.doa_limits = {
            "urea": 65.0,
            "tsp": 30.0,
            "mop": 30.0,
        }

        # Sri Lankan banned/restricted pesticides list
        self.banned_chemicals = [
            "methyl parathion", "paraquat", "carbofuran", "methamidophos",
            "monocrotophos", "phosphamidon", "endosulfan"
        ]

        self.allergen_watchlist = {
            "chlorpyrifos": "respiratory sensitizer",
            "mancozeb": "skin irritant",
            "glyphosate": "suspected carcinogen (WHO Group 2A)"
        }

    def verify_fertilizer_dosage(self, rec: FertilizerRecommendation) -> List[SafetyVerdict]:
        """Check fertilizer dosages against DoA maximum limits."""
        verdicts = []
        checks = [
            ("urea", rec.urea_dosage_per_acre_kg),
            ("tsp", rec.tsp_dosage_per_acre_kg),
            ("mop", rec.mop_dosage_per_acre_kg),
        ]
        for chem, dose in checks:
            limit = self.doa_limits.get(chem, float("inf"))
            if dose > limit:
                verdicts.append(SafetyVerdict(
                    check_name=f"{chem}_dosage",
                    passed=False,
                    message=f"Dosage {dose} kg/ac exceeds DoA max of {limit} kg/ac",
                    severity="warning"
                ))
            else:
                verdicts.append(SafetyVerdict(
                    check_name=f"{chem}_dosage",
                    passed=True,
                    message=f"Dosage {dose} kg/ac within DoA limit of {limit} kg/ac",
                    severity="info"
                ))
        return verdicts

    def verify_banned_chemicals(self, treatment_list: List[str]) -> List[SafetyVerdict]:
        """Cross-check recommended treatments against banned pesticides list."""
        verdicts = []
        for treatment in treatment_list:
            for banned in self.banned_chemicals:
                if banned.lower() in treatment.lower():
                    verdicts.append(SafetyVerdict(
                        check_name="banned_chemical",
                        passed=False,
                        message=f"'{banned}' is RESTRICTED under Sri Lankan pesticide regulations",
                        severity="critical"
                    ))
        if not verdicts:
            verdicts.append(SafetyVerdict(
                check_name="banned_chemical",
                passed=True,
                message="No banned pesticides detected in recommendations",
                severity="info"
            ))
        return verdicts

    def verify_allergens(self, treatment_list: List[str]) -> List[SafetyVerdict]:
        """Check for known allergens in recommended treatments."""
        verdicts = []
        for treatment in treatment_list:
            for chem, risk in self.allergen_watchlist.items():
                if chem.lower() in treatment.lower():
                    verdicts.append(SafetyVerdict(
                        check_name="allergen_screen",
                        passed=False,
                        message=f"'{chem}' identified as {risk} - include safety precautions",
                        severity="warning"
                    ))
        if not verdicts:
            verdicts.append(SafetyVerdict(
                check_name="allergen_screen",
                passed=True,
                message="No known allergen risks detected",
                severity="info"
            ))
        return verdicts

    def get_citations(self) -> List[str]:
        """Return relevant Sri Lankan regulatory citations."""
        return [
            "Department of Agriculture Sri Lanka - Pesticide Act No. 33 of 1980",
            "Department of Agriculture Sri Lanka - Fertilizer Ordinance No. 1 of 1995",
            "Sri Lanka Standards Institution - SLS 1164:2019 Fertilizer Specification",
            "FAO Sri Lanka - Code of Conduct for Pesticide Management (2023)",
        ]

    def process(self, diagnostic: Optional[DiagnosticResult] = None, fertilizer: Optional[FertilizerRecommendation] = None) -> ReflectionResult:
        """Run all safety checks on agent outputs."""
        all_verdicts: List[SafetyVerdict] = []
        warnings: List[str] = []
        biosecurity_alerts: List[str] = []

        # Check fertilizer dosage
        if fertilizer:
            all_verdicts.extend(self.verify_fertilizer_dosage(fertilizer))

        # Check diagnostic treatments
        if diagnostic:
            all_verdicts.extend(self.verify_banned_chemicals(diagnostic.treatment_recommended))
            all_verdicts.extend(self.verify_allergens(diagnostic.treatment_recommended))

        # Aggregate warnings
        for v in all_verdicts:
            if not v.passed:
                warnings.append(v.message)
                if v.severity == "critical":
                    biosecurity_alerts.append(f"BIOSECURITY: {v.message}")

        all_passed = all(v.passed for v in all_verdicts)

        return ReflectionResult(
            recommendation_id=f"ref_{uuid.uuid4().hex[:8]}",
            all_checks_passed=all_passed,
            verdicts=all_verdicts,
            warnings=warnings,
            regulatory_citations=self.get_citations(),
            biosecurity_alerts=biosecurity_alerts
        )
