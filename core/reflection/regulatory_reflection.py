"""
RegulatoryReflection Module - Version 3.0 Architecture
Audits recommended chemical treatments against Sri Lanka Pesticide Act No. 33 (WHO Class Ia/Ib bans).
"""

import re
from typing import Dict, Any, List, Tuple


class RegulatoryReflection:
    """Conditional safety reflection layer enforcing Pesticide Act No. 33 compliance."""

    BANNED_CHEMICALS = [
        "paraquat", "carbofuran", "endosulfan", "glyphosate", "monocrotophos", "methamidophos"
    ]

    @classmethod
    def audit_response(cls, response_text: str) -> Tuple[str, bool]:
        """
        Scans response text for banned WHO Class Ia/Ib chemicals.
        Returns (audited_text, contains_violation).
        """
        text_lower = response_text.lower()
        violations_found = []

        for chem in cls.BANNED_CHEMICALS:
            if chem in text_lower:
                violations_found.append(chem)

        if not violations_found:
            return response_text, False

        # Apply chemical safety guard
        warning_banner = (
            "\n\n> ⚠️ **REGULATORY SAFETY WARNING (Pesticide Act No. 33 Audit)**\n"
            f"> Banned chemical(s) detected: {', '.join(violations_found).title()}.\n"
            "> These chemicals are strictly prohibited under Sri Lankan DOA regulations. "
            "Please use only DOA approved systemic fungicides (e.g. Tricyclazole 75% WP, Tebuconazole).\n"
        )
        return response_text + warning_banner, True
