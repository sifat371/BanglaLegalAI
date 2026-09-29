"""Claim-level support assessment for cited legal answers.

The orchestration is deterministic: answer claims are segmented, citations are
bound to retrieved documents, and only the cited source text is sent to the
support evaluator. The default evaluator is model-based and therefore remains
experimental until benchmarked on reviewed Bangladesh-law examples.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from langchain_core.documents import Document

_CITATION = re.compile(r"\\[S(?P<number>[1-9]\\d*)\\]")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।])\\s+|\\n+")


class SupportEvaluator(Protocol):
    """Pluggable evaluator for one claim against its cited source passages."""

    name: str

    def evaluate(
        self,
        claim: str,
        cited_sources: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Return label/reason/evidence for one claim."""


class LLMClaimSupportEvaluator:
    """Conservative source-only support classifier backed by an LLM."""

    name = "llm_source_only_v1"
    _allowed_labels = {"supported", "contradicted", "insufficient"}

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    @staticmethod
    def _message_text(message: Any) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\\n".join(parts)
        return str(content)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        rendered = text.strip()
        fence = chr(96) * 3
        if rendered.startswith(fence):
            rendered = re.sub(
                r"^" + re.escape(fence) + r"(?:json)?\\s*",
                "",
                rendered,
                flags=re.IGNORECASE,
            )
            rendered = re.sub(r"\\s*" + re.escape(fence) + r"$", "", rendered)
        try:
            value = json.loads(rendered)
        except json.JSONDecodeError:
            start = rendered.find("{")
            end = rendered.rfind("}")
            if start < 0 or end <= start:
                raise
            value = json.loads(rendered[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("support evaluator response must be a JSON object")
        return value

    def evaluate(
        self,
        claim: str,
        cited_sources: list[dict[str, str]],
    ) -> dict[str, Any]:
        sources = "\\n\\n".join(
            f"[{source['source_id']}]\\n{source['text']}"
            for source in cited_sources
        )
        prompt = f"""You are checking whether a legal-answer claim is supported
by the cited source text.

Use ONLY the source passages below. Do not use memory, outside law,
assumptions, or unstated facts.

Labels:
- supported: all material parts of the claim are explicitly supported or
  directly entailed by the cited text.
- contradicted: the cited text directly conflicts with a material part of
  the claim.
- insufficient: the cited text does not establish the full claim, is
  ambiguous, or lacks needed information.

Be conservative. If only part of a compound claim is supported, choose insufficient.
Do not decide whether the law is current unless the cited text itself establishes that.

Claim:
{claim}

Cited sources:
{sources}

Return JSON only:
{{"label":"supported|contradicted|insufficient",
  "reason":"brief source-bound reason",
  "evidence":"brief supporting/conflicting passage or empty string"}}
"""
        try:
            parsed = self._extract_json(self._message_text(self.llm.invoke(prompt)))
            label = str(parsed.get("label", "")).strip().lower()
            if label not in self._allowed_labels:
                raise ValueError(f"invalid support label: {label!r}")
            return {
                "label": label,
                "reason": str(parsed.get("reason", "")).strip()[:600],
                "evidence": str(parsed.get("evidence", "")).strip()[:600],
                "evaluator_error": None,
            }
        except Exception as exc:
            return {
                "label": "insufficient",
                "reason": "Support evaluator could not produce a valid source-bound judgment.",
                "evidence": "",
                "evaluator_error": f"{type(exc).__name__}: {exc}",
            }


class ClaimSupportVerifier:
    """Bind cited claims to source passages and assess support claim by claim."""

    def __init__(self, evaluator: SupportEvaluator, *, max_claims: int = 12) -> None:
        if max_claims < 1:
            raise ValueError("max_claims must be >= 1")
        self.evaluator = evaluator
        self.max_claims = max_claims

    @staticmethod
    def _source_id(number: str) -> str:
        return f"S{int(number)}"

    @classmethod
    def _normalized_segments(cls, answer: str) -> list[str]:
        # Normalize the common "claim. [S1]" form to "claim [S1]." before
        # sentence splitting so the source marker stays attached to its claim.
        normalized = re.sub(
            r"([.!?।])\s*((?:\[S[1-9]\d*\]\s*)+)",
            r" \2\1",
            answer,
        )
        return [
            segment.strip()
            for segment in _SENTENCE_BOUNDARY.split(normalized)
            if segment.strip()
        ]

    @classmethod
    def extract_cited_claims(cls, answer: str) -> list[dict[str, Any]]:
        claims = []
        claim_number = 0

        for original in cls._normalized_segments(answer):
            source_ids = [
                cls._source_id(match.group("number"))
                for match in _CITATION.finditer(original)
            ]
            source_ids = list(dict.fromkeys(source_ids))
            if not source_ids:
                continue

            claim_text = _CITATION.sub("", original).strip()
            claim_text = re.sub(r"\s+", " ", claim_text)
            claim_text = re.sub(r"\s+([,.;:!?।])", r"\1", claim_text)
            if not claim_text:
                continue

            claim_number += 1
            claims.append(
                {
                    "claim_id": f"C{claim_number}",
                    "text": claim_text,
                    "source_ids": source_ids,
                    "original": original,
                }
            )

        return claims

    @classmethod
    def extract_uncited_segments(cls, answer: str) -> list[str]:
        """Return nontrivial answer segments that have no canonical source marker.

        These are coverage warnings only. They are not automatically classified
        as factual legal claims.
        """
        uncited = []
        for segment in cls._normalized_segments(answer):
            if _CITATION.search(segment):
                continue

            cleaned = re.sub(r"^[#>*\-\d.)\s]+", "", segment).strip()
            if len(cleaned) < 20:
                continue
            if cleaned.endswith(":") and len(cleaned.split()) <= 8:
                continue
            uncited.append(cleaned)

        return uncited

    def verify(
        self,
        answer: str,
        documents: list[Document],
        sources: list[dict[str, Any]],
        citation_verification: dict[str, Any],
    ) -> dict[str, Any]:
        citation_status = citation_verification.get("status")
        if citation_status != "verified":
            return {
                "status": "not_run",
                "experimental": True,
                "evaluator": self.evaluator.name,
                "claims_total": 0,
                "claims_evaluated": 0,
                "counts": {},
                "claims": [],
                "truncated": False,
                "independently_validated": False,
                "message": (
                    "Claim support assessment requires structurally valid answer citations."
                ),
            }

        source_text_by_id = {
            str(source.get("source_id")): document.page_content
            for source, document in zip(sources, documents, strict=True)
            if source.get("source_id")
        }
        source_meta_by_id = {
            str(source.get("source_id")): source
            for source in sources
            if source.get("source_id")
        }

        extracted = self.extract_cited_claims(answer)
        uncited_segments = self.extract_uncited_segments(answer)
        selected = extracted[: self.max_claims]
        results = []

        for claim in selected:
            cited_sources = []
            missing_sources = []
            for source_id in claim["source_ids"]:
                source_text = source_text_by_id.get(source_id)
                if source_text is None:
                    missing_sources.append(source_id)
                    continue
                cited_sources.append(
                    {
                        "source_id": source_id,
                        "text": source_text,
                    }
                )

            if missing_sources or not cited_sources:
                assessment = {
                    "label": "insufficient",
                    "reason": (
                        "One or more cited source passages were unavailable "
                        "for verification."
                    ),
                    "evidence": "",
                    "evaluator_error": None,
                }
            else:
                assessment = self.evaluator.evaluate(claim["text"], cited_sources)

            provenance = []
            for source_id in claim["source_ids"]:
                source = source_meta_by_id.get(source_id, {})
                provenance.append(
                    {
                        "source_id": source_id,
                        "citation": source.get("citation", ""),
                        "document_id": source.get("document_id", ""),
                        "chunk_id": source.get("chunk_id", ""),
                        "page": source.get("page", ""),
                    }
                )

            results.append(
                {
                    **claim,
                    **assessment,
                    "provenance": provenance,
                }
            )

        counts = {
            label: sum(1 for result in results if result["label"] == label)
            for label in ("supported", "contradicted", "insufficient")
        }

        if not results:
            status = "not_applicable"
        elif counts["contradicted"]:
            status = "contradicted"
        elif counts["insufficient"]:
            status = "insufficient"
        else:
            status = "supported"

        return {
            "status": status,
            "experimental": True,
            "evaluator": self.evaluator.name,
            "claims_total": len(extracted),
            "claims_evaluated": len(results),
            "counts": counts,
            "claims": results,
            "truncated": len(extracted) > len(selected),
            "uncited_segments": uncited_segments[:10],
            "uncited_segments_count": len(uncited_segments),
            "coverage_complete": not uncited_segments,
            "independently_validated": False,
            "message": self._message(status),
        }

    @staticmethod
    def _message(status: str) -> str:
        messages = {
            "supported": (
                "The experimental verifier assessed every evaluated cited claim as supported "
                "by its cited retrieved passage."
            ),
            "contradicted": (
                "At least one evaluated claim was assessed as contradicted by its cited passage."
            ),
            "insufficient": (
                "At least one evaluated claim was not fully established by its cited passage."
            ),
            "not_applicable": "No cited answer claims were available for support assessment.",
        }
        return messages[status]
