"""
tests/test_streamlit_app.py — drives the real Streamlit app headlessly with
Streamlit's AppTest, so UI wiring breaks (a renamed field, a panel calling
a changed function) fail CI instead of being found by hand.

Each test generates a small simulated dataset through the app's own form,
runs the analysis, and checks what the page shows. The CSV upload path is
covered at the function level in tests/test_upload.py (AppTest cannot
drive st.file_uploader).
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from streamlit.testing.v1 import AppTest

os.environ.setdefault("MPLBACKEND", "Agg")

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "streamlit_app.py")
TIMEOUT = 120


def _button(at: AppTest, label_fragment: str):
    return next(b for b in at.button if label_fragment in b.label)


def _generate(at: AppTest, n_users: int = 2000, **checkboxes: bool) -> None:
    """Fill in the generate form (small n keeps the test fast) and submit."""
    next(n for n in at.number_input if n.label == "Number of users").set_value(n_users)
    for label_fragment, value in checkboxes.items():
        box = next(c for c in at.checkbox if label_fragment in c.label)
        box.check() if value else box.uncheck()
    _button(at, "Generate data").click().run()


def _assert_no_failures(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not [e for e in at.error if "Analysis failed" in e.value]


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()
    return at


def test_default_analysis_renders_every_panel(app):
    _generate(app)
    _button(app, "Run analysis").click().run()

    _assert_no_failures(app)
    subheaders = [h.value for h in app.subheader]
    for panel in ("Validity Checks", "Effect Estimate",
                  "Sequential Testing / Peeking Risk", "Segment Analysis"):
        assert panel in subheaders
    assert any("Randomization looks healthy" in s.value for s in app.success)
    assert any(m.label == "SRM p-value" for m in app.metric)


def test_srm_bug_shows_blocking_warning(app):
    corrupted = next(c for c in app.checkbox if "broken randomization" in c.label)
    corrupted.check().run()
    next(s for s in app.slider if "Corrupted treatment split" in s.label).set_value(0.40)
    _generate(app, n_users=4000)
    _button(app, "Run analysis").click().run()

    _assert_no_failures(app)
    assert any("Do not trust this result" in e.value for e in app.error)


def test_multi_arm_revenue_ratio_metric_with_segments(app):
    _generate(app, **{"second treatment arm": True})
    next(s for s in app.selectbox if s.label == "Metric").set_value(
        "revenue_per_conversion"
    ).run()
    app.multiselect[0].set_value(["device_type"]).run()
    _button(app, "Run analysis").click().run()

    _assert_no_failures(app)
    arm_table = next(df.value for df in app.dataframe if "Arm" in df.value.columns)
    assert list(arm_table["Arm"]) == ["treatment", "treatment_b"]
    assert set(arm_table["Estimator"]) == {"cuped_delta_ratio"}
    segment_headers = [m.value for m in app.markdown if m.value.startswith("**By")]
    assert len(segment_headers) == 2  # one per arm
    assert any("Cochran's Q" in m.value for m in app.markdown)


def test_peeking_check_logs_checkpoint_and_stores_schedule(app):
    _generate(app)
    next(c for c in app.checkbox if "peeking" in c.label).check().run()
    _button(app, "Run analysis").click().run()
    _button(app, "Log this checkpoint").click().run()
    _button(app, "Run analysis").click().run()

    _assert_no_failures(app)
    assert any("conclusions" in m.value for m in (*app.success, *app.error))

    engine = app.session_state["engine"]
    experiment_id = app.session_state["experiment_id"]
    with engine.connect() as conn:
        checkpoints = conn.execute(
            text("SELECT COUNT(*) FROM sequential_checkpoints WHERE experiment_id = :eid"),
            {"eid": experiment_id},
        ).scalar()
        schedule = conn.execute(
            text("SELECT n_checkpoints_planned, checkpoint_n FROM experiments "
                 "WHERE experiment_id = :eid"),
            {"eid": experiment_id},
        ).fetchone()
    assert checkpoints == 1
    assert tuple(schedule) == (10, 400)
