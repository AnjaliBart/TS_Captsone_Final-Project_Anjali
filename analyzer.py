"""
Modules description:
  1. Claim Extractor   — extracts promises from product description
  2. Review Verifier   — confirms / contradicts / flags unverified claims
  3. Return Reason ML  — classifies return reason per review (uses trained model)
  4. Risk Scorer       — aggregates into 0-100 return risk score
  5. Fix Generator     — produces actionable description fixes
"""

import re, pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

MODEL_PATH = "model/return_classifier.pkl"

#  Claim patterns for Apparel & Beauty 
CLAIM_PATTERNS = {
    "Size / Fit":         r"\b(true to size|fits? (well|perfectly|snugly)|runs (small|large|true)|adjustable|one[- ]size|size(able)?)\b",
    "Material Quality":   r"\b(premium|high[- ]quality|durable|soft|breathable|lightweight|100%\s*\w+|genuine leather|cotton|polyester|wool|silk|linen)\b",
    "Colour Accuracy":    r"\b(vibrant|colourfast|as (shown|pictured|described)|true colour|accurate colour)\b",
    "Skin-Safe / Hypoallergenic": r"\b(hypoallergenic|dermatologist tested|skin[- ]safe|non[- ]irritating|paraben[- ]free|cruelty[- ]free|allergy tested)\b",
    "Long-Lasting":       r"\b(long[- ]lasting|all[- ]day|24[- ]hour|waterproof|sweat[- ]proof|smudge[- ]proof|transfer[- ]proof)\b",
    "Fragrance / Scent":  r"\b(fragrance[- ]free|no scent|pleasant scent|mild fragrance|natural scent)\b",
    "Packaging":          r"\b(gift[- ]wrap(ped)?|premium packaging|eco[- ]friendly pack|securely packed|luxury packaging)\b",
    "Warranty / Returns": r"\b(\d+[- ]?(year|month|day)[- ]warranty|lifetime warranty|money[- ]back|satisfaction guaranteed|free returns)\b",
    "Value for Money":    r"\b(value for money|affordable|budget[- ]friendly|cost[- ]effective|best price|great deal)\b",
    "Comfort":            r"\b(comfortable|soft touch|gentle on skin|smooth finish|cosy|cozy|ergonomic)\b",
}

POSITIVE_SIGNALS = [
    "exactly as described","as expected","true to size","love it","great quality",
    "highly recommend","perfect","excellent","amazing","well made","comfortable",
    "fits perfectly","beautiful","long lasting","smells great","soft","gentle",
]
NEGATIVE_SIGNALS = [
    "not as described","misleading","poor quality","cheap","broke","wrong size",
    "doesn't fit","waste of money","disappointed","return","refund","fake",
    "not worth","bad quality","falls apart","not what i expected",
    "false advertising","looks nothing like","irritated my skin","caused breakout",
    "allergic reaction","color faded","wrong shade","runs small","runs large",
]

from preprocess import RETURN_REASONS, NO_RETURN_LABEL, label_return_reason


@dataclass
class ClaimVerdict:
    claim_type:  str
    claim_text:  str
    status:      str          # Confirmed | Contradicted | Unverified
    evidence:    list[str]
    confidence:  float


@dataclass
class AnalysisResult:
    product_description:       str
    reviews:                   list[str]
    claim_verdicts:            list[ClaimVerdict]
    return_reason_distribution: dict[str, float]
    top_return_reason:         str
    risk_score:                float
    risk_level:                str
    risk_breakdown:            dict[str, float]
    suggested_fixes:           list[str]


class ReturnAnalyzer:

    def __init__(self):
        self._clf    = None
        self._labels = None

    def _load_model(self):
        if self._clf is None:
            try:
                with open(MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                self._clf    = data["pipeline"]
                self._labels = data["labels"]
            except FileNotFoundError:
                self._clf    = None
                self._labels = []

    #  Module 1: Claim Extractor 
    def extract_claims(self, description: str) -> dict[str, str]:
        found = {}
        desc_lower = description.lower()
        for claim_type, pattern in CLAIM_PATTERNS.items():
            match = re.search(pattern, desc_lower, re.IGNORECASE)
            if match:
                found[claim_type] = match.group(0)
        if not found:
            # fallback: grab capitalised noun phrases as generic claims
            for w in re.findall(r"\b[A-Z][a-z]{3,}(?:\s[a-z]+)?\b", description)[:5]:
                found[f"Feature: {w}"] = w
        return found

    # Module 2: Review Verifier 
    def verify_claim(self, claim_type, claim_text, reviews):
        pos, neg, evidence = 0, 0, []
        kws = claim_text.lower().split()
        for review in reviews:
            rev = review.lower()
            if not any(kw in rev for kw in kws):
                continue
            p = sum(1 for s in POSITIVE_SIGNALS if s in rev)
            n = sum(1 for s in NEGATIVE_SIGNALS if s in rev)
            if p > n:
                pos += 1
                if len(evidence) < 2:
                    evidence.append(f"{review[:120]}...")
            elif n > p:
                neg += 1
                if len(evidence) < 2:
                    evidence.append(f"{review[:120]}...")
        total = pos + neg
        if total == 0:
            return ClaimVerdict(claim_type, claim_text, "Unverified", [], 0.5)
        conf   = max(pos, neg) / total
        status = "Confirmed" if pos >= neg else "Contradicted"
        return ClaimVerdict(claim_type, claim_text, status, evidence, conf)

    #  Module 3: Return Reason Predictor 
    def predict_return_reasons(self, reviews: list[str]) -> dict[str, float]:
        self._load_model()
        if self._clf is not None:
            cleaned = [re.sub(r"[^a-z0-9\s]", " ", r.lower()) for r in reviews]
            probs   = self._clf.predict_proba(cleaned)
            scores  = {lbl: 0.0 for lbl in self._labels}
            for row in probs:
                for lbl, p in zip(self._labels, row):
                    scores[lbl] += p
            total = sum(scores.values())
            return {k: v/total for k, v in scores.items()} if total else scores
        else:
            # keyword fallback if model not trained yet
            counts = {r: 0 for r in list(RETURN_REASONS.keys())}
            for rev in reviews:
                reason = label_return_reason(rev)
                if reason != NO_RETURN_LABEL:
                    counts[reason] = counts.get(reason, 0) + 1
            total = sum(counts.values()) or 1
            return {k: v/total for k, v in counts.items()}

    #  Module 4: Risk Scorer 
    def compute_risk(self, verdicts, reason_dist, reviews):
        # 1. Claim contradiction score (0–40)
        if verdicts:
            contradicted = sum(1 for v in verdicts if v.status == "Contradicted")
            unverified   = sum(1 for v in verdicts if v.status == "Unverified")
            claim_score  = min((contradicted*40 + unverified*15) / len(verdicts), 40)
        else:
            claim_score = 20

        # 2. Return reason signal (0-35)
        weights = {
            "Expectation Mismatch":   1.0,
            "Wrong / Missing Item":   0.9,
            "Damaged / Defective":    0.85,
            "Quality Issue":          0.75,
            "Size / Fit Issue":       0.65,
            "Delivery / Packaging":   0.4,
        }
        reason_score = min(sum(reason_dist.get(r,0)*w*35 for r, w in weights.items()), 35)

        # 3. Negative sentiment density (0-25)
        neg_ratio    = sum(1 for r in reviews if any(s in r.lower() for s in NEGATIVE_SIGNALS)) / max(len(reviews),1)
        sent_score   = neg_ratio * 25

        total = claim_score + reason_score + sent_score
        breakdown = {
            "Claim Contradictions": round(claim_score, 1),
            "Return Reason Signal": round(reason_score, 1),
            "Negative Sentiment":   round(sent_score,   1),
        }
        return round(min(total, 100), 1), breakdown

    @staticmethod
    def risk_level(score: float) -> str:
        if score < 25: return "Low"
        if score < 50: return "Medium"
        if score < 75: return "High"
        return "Critical"

    #  Module 5: Fix Generator 
    def generate_fixes(self, verdicts, top_reason):
        fixes = []
        for v in verdicts:
            if v.status == "Contradicted":
                fixes.append(
                    f" Remove or qualify the **{v.claim_type}** claim (\"{v.claim_text}\") "
                    f"— customer reviews contradict it."
                )
            elif v.status == "Unverified":
                fixes.append(
                    f"Add proof or specifics for the **{v.claim_type}** claim "
                    f"(\"{v.claim_text}\") — no reviews validate it."
                )
        tip_map = {
            "Size / Fit Issue":       "Add a detailed size chart. Mention if the product runs small or large.",
            "Expectation Mismatch":   "Add real customer photos and align description with actual product.",
            "Quality Issue":          "Specify exact materials (e.g. '100% Egyptian cotton') and quality checks.",
            "Damaged / Defective":    "Describe packaging method and QC process in the listing.",
            "Wrong / Missing Item":   "Clarify variant details (colour, size) clearly on the product page.",
            "Delivery / Packaging":   "Describe packaging standards and expected delivery timeline.",
        }
        if top_reason in tip_map:
            fixes.append(tip_map[top_reason])
        if not fixes:
            fixes.append("Description appears accurate. Monitor review trends regularly.")
        return fixes

    #  Main entry point 
    def analyze(self, description: str, reviews: list[str]) -> AnalysisResult:
        claims   = self.extract_claims(description)
        verdicts = [self.verify_claim(ct, cv, reviews) for ct, cv in claims.items()]
        reason_dist  = self.predict_return_reasons(reviews)
        top_reason   = max(reason_dist, key=reason_dist.get) if reason_dist else "Unknown"
        risk_score, breakdown = self.compute_risk(verdicts, reason_dist, reviews)
        level  = self.risk_level(risk_score)
        fixes  = self.generate_fixes(verdicts, top_reason)
        return AnalysisResult(
            product_description=description, reviews=reviews,
            claim_verdicts=verdicts, return_reason_distribution=reason_dist,
            top_return_reason=top_reason, risk_score=risk_score,
            risk_level=level, risk_breakdown=breakdown, suggested_fixes=fixes,
        )


_analyzer = None
def get_analyzer() -> ReturnAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = ReturnAnalyzer()
    return _analyzer
