# A/B Test Analysis Engine

Structural enforcement of correct experimentation methodology — SRM detection, power
analysis, CUPED variance reduction, empirical peeking-inflation correction, and
BH-corrected segment heterogeneity — validated against synthetic data with known
ground truth, not asserted.

> Status: **under construction.**

## Setup

\`\`\`bash
git clone <repo_url>
cd ab-test-engine
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest -m "not slow"
\`\`\`

## Why this exists

Naive `p < 0.05` A/B testing fails in four specific ways: underpowered tests,
peeking-induced false-positive inflation, sample ratio mismatch, and
Simpson's-paradox-masked heterogeneity. This engine makes the relevant checks
mandatory and ordered rather than optional.

## Validation evidence

_(populated after Phase 4/5 with real numbers — CI coverage %, CUPED variance
reduction %, empirical peeking FPR before/after correction)_