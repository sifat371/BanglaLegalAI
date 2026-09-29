"""Grounding gate and repair feedback for generated legal answers."""

from __future__ import annotations

from typing import Any


class GroundingEnforcer:
    """Apply conservative fail-closed rules to answer verification results.

    Passing this gate means the answer passed the configured citation and
    experimental support checks. It does not establish independent legal
    correctness.
    """

    @staticmethod
    def evaluate(
        citation_verification: dict[str, Any],
        claim_support_verification: dict[str, Any],
    ) -> dict[str, Any]:
        reasons: list[str] = []

        citation_status = citation_verification.get("status")
        if citation_status != "verified":
            reasons.append(f"citation_integrity:{citation_status or 'missing'}")

        support_status = claim_support_verification.get("status")
        if support_status != "supported":
            reasons.append(f"claim_support:{support_status or 'missing'}")

        if claim_support_verification.get("coverage_complete") is not True:
            reasons.append("semantic_coverage:incomplete_or_unknown")

        if claim_support_verification.get("truncated") is not False:
            reasons.append("semantic_coverage:truncated_or_unknown")

        evaluator_errors = [
            claim.get("claim_id", "unknown")
            for claim in claim_support_verification.get("claims", [])
            if claim.get("evaluator_error")
        ]
        if evaluator_errors:
            reasons.append(
                "evaluator_error:" + ",".join(evaluator_errors)
            )

        return {
            "acceptable": not reasons,
            "reasons": reasons,
        }

    @staticmethod
    def repair_feedback(
        citation_verification: dict[str, Any],
        claim_support_verification: dict[str, Any],
    ) -> str:
        """Render only the verification failures needed by the repair model."""
        lines: list[str] = []

        citation_status = citation_verification.get("status")
        if citation_status != "verified":
            lines.append(f"Citation integrity status: {citation_status}.")
            invalid = citation_verification.get("invalid_source_ids", [])
            if invalid:
                lines.append(
                    "Unknown citation IDs: " + ", ".join(invalid) + "."
                )
            noncanonical = citation_verification.get("noncanonical_markers", [])
            if noncanonical:
                lines.append(
                    "Noncanonical citation markers: "
                    + ", ".join(noncanonical)
                    + "."
                )

        for claim in claim_support_verification.get("claims", []):
            label = claim.get("label")
            if label not in {"contradicted", "insufficient"}:
                continue
            source_ids = ", ".join(claim.get("source_ids", [])) or "none"
            lines.append(
                f"{claim.get('claim_id', '?')} is {label}: "
                f"{claim.get('text', '')} "
                f"(sources: {source_ids}). "
                f"Reason: {claim.get('reason', '')}"
            )

        uncited = claim_support_verification.get("uncited_segments", [])
        if uncited:
            lines.append(
                "Nontrivial uncited answer segments that were not checked:"
            )
            lines.extend(f"- {segment}" for segment in uncited)

        if claim_support_verification.get("truncated"):
            lines.append(
                "The answer contained more cited claims than the verifier "
                "was configured to check. Make the repaired answer shorter."
            )

        if any(
            claim.get("evaluator_error")
            for claim in claim_support_verification.get("claims", [])
        ):
            lines.append(
                "At least one support-evaluator call failed. Prefer a shorter, "
                "simpler answer with explicit source-backed claims."
            )

        return "\n".join(lines) or "No specific repair feedback was available."

    @staticmethod
    def blocked_answer() -> str:
        """Return a deterministic non-substantive fallback."""
        return (
            "I couldn't produce a fully source-supported answer from the "
            "retrieved legal material. Please review the listed sources or "
            "refine the question so the answer can be grounded more precisely."
        )
