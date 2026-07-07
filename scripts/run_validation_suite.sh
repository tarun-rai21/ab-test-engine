#!/usr/bin/env bash
# scripts/run_validation_suite.sh
#
# Runs the slow, simulation-based validation harness and writes real
# observed numbers to validation/validation_report.md.
#
# This is a MANUAL release gate (Phase 0 CI design), not run on every push —
# these tests take minutes; running them per-commit would slow CI enough
# that it stops being trusted/checked, per the reasoning established when
# @pytest.mark.slow was first introduced.
#
# Usage: bash scripts/run_validation_suite.sh

set -e

echo "Running ground-truth validation harness (slow — several minutes)..."

pytest validation/ -v -s -m slow --tb=short | tee /tmp/validation_output.txt

echo ""
echo "Validation run complete. Update validation/validation_report.md manually"
echo "with the observed numbers above before tagging a release."