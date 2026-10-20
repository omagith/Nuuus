"""The self-test reports every deep check and fails the exit code on errors."""
from app.core import self_test


def test_self_test_report_and_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(self_test, "REPORT_FILE", tmp_path / "self_test.txt")
    monkeypatch.setattr("app.core.diagnostics.run_diagnostics", lambda: [("ok", "GPU", "fake")])

    def boom():
        raise RuntimeError("сломалось")

    monkeypatch.setattr(self_test, "DEEP_CHECKS", [("Хорошая", lambda: "работает"), ("Плохая", boom)])
    assert self_test.run_self_test() == 1
    report = (tmp_path / "self_test.txt").read_text(encoding="utf-8")
    assert "✓ GPU: fake" in report and "✓ Хорошая: работает" in report
    assert "✗ Плохая: RuntimeError: сломалось" in report and "ошибок — 1" in report

    monkeypatch.setattr(self_test, "DEEP_CHECKS", [("Хорошая", lambda: "работает")])
    assert self_test.run_self_test() == 0
