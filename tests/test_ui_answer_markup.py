"""Finished answers lose stray markdown/LaTeX markers but keep emphasis."""
import pytest
from PyQt6.QtCore import Qt

from app.ui import instant_tab as it
from tests.test_ui_support import dispose, light_theme, qapp  # noqa: F401  (fixtures)


@pytest.mark.parametrize("raw, clean", [
    ("## Ответ", "Ответ"),
    ("- первый пункт", "• первый пункт"),
    ("Это `код`", "Это код"),
    ("Ответ: $\\frac{a}{b}$", "Ответ: a/b"),
    ("$\\Delta = k\\lambda$", "Δ = kλ"),
    ("\\(x^{2} \\cdot y\\)", "x^2 · y"),
    ("$\\sqrt{2} \\approx 1{,}41$", "√(2) ≈ 1,41"),
    ("Обычный текст, 2 * 3 = 6.", "Обычный текст, 2 * 3 = 6."),
])
def test_clean_answer_line(raw, clean):
    assert it.clean_answer_line(raw) == clean


def test_answer_markdown_is_cleaned_after_done(light_theme):
    """The answer card streams raw text, then shows it cleaned; *italics*
    survive as rich text (the whole answer is bold anyway)."""
    panel = it.AnswersPanel()
    try:
        panel.add_question(1, "Что такое предел?", typed=False)
        for tok in ["**Предел** ", "функции — это $L$, ", "к которому *стремится* f(x).\n", "- и всё"]:
            panel.append_token(1, tok)
        assert "**" in panel.card_text(1)  # raw while streaming
        panel.finish_answer(1, "")
        assert panel.card_text(1) == "Предел функции — это L, к которому стремится f(x).\n• и всё"
        body = panel.card(1).body
        assert body.textFormat() == Qt.TextFormat.RichText
        assert "<i>стремится</i>" in body.text() and "**" not in body.text()
        assert panel.open_cards() == []
    finally:
        dispose(panel)


def test_plain_answer_stays_plain_text(light_theme):
    panel = it.AnswersPanel()
    try:
        panel.add_question(1, "Сколько?", typed=True)
        panel.append_token(1, "Два <b>не</b> тег")
        panel.finish_answer(1, "")
        body = panel.card(1).body
        assert body.textFormat() == Qt.TextFormat.PlainText and body.text() == "Два <b>не</b> тег"
    finally:
        dispose(panel)
