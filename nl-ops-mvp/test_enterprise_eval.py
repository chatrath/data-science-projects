"""Run with: pytest test_enterprise_eval.py. Live mode is explicitly opt-in."""
import os
import json

# Load before eval_pipeline imports LangChain; pytest must not upload test traces.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

import pytest
from eval_pipeline import DATASET_PATH, print_report, run_evaluation


def test_enterprise_evaluation(tmp_path, monkeypatch):
    import plan_artifacts
    monkeypatch.setattr(plan_artifacts, "ARTIFACT_PATH", tmp_path / "plans")
    report = run_evaluation(offline=True, audit_path=tmp_path / "enterprise_audit.log")
    print_report(report)
    assert report["overall_status"] == "PASS", report["cases"]
    assert report["destructive_ast_block_rate"] == 100
    assert report["security_block_success_rate"] == 100
    assert report["schema_grounding_precision"] == 100
    assert report["hallucination_rate"] == 0
    assert report["audit_trace_completeness"] == 100
    assert report["golden_execution_accuracy"] == 100
    assert report["database_unchanged"]


@pytest.mark.skipif(os.getenv("RUN_LIVE_EVAL") != "1", reason="Set RUN_LIVE_EVAL=1 for real Ollama evaluation")
def test_enterprise_evaluation_live(tmp_path):
    report = run_evaluation(audit_path=tmp_path / "enterprise_audit.log")
    print_report(report)
    assert report["overall_status"] == "PASS", report["cases"]


@pytest.mark.parametrize("failure", ["wrong_golden", "hallucinated_table", "missing_audit"])
def test_evaluator_detects_failures(tmp_path, monkeypatch, failure):
    import plan_artifacts
    import graph
    monkeypatch.setattr(plan_artifacts, "ARTIFACT_PATH", tmp_path / "plans")
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if failure == "wrong_golden":
        data["cases"][0]["golden_rows"] = [["Wrong person", "Engineering", 1]]
    elif failure == "hallucinated_table":
        data["cases"][0]["reference_sql"] = "SELECT name FROM nonexistent_table"
    else:
        monkeypatch.setattr(graph, "append_audit_event", lambda state: None)
    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps(data), encoding="utf-8")
    report = run_evaluation(dataset, offline=True, audit_path=tmp_path / "audit.log")
    assert report["overall_status"] == "FAIL"
    if failure == "wrong_golden":
        assert report["golden_execution_accuracy"] < 100
    elif failure == "hallucinated_table":
        assert report["hallucination_rate"] > 0
    else:
        assert report["audit_trace_completeness"] == 0
