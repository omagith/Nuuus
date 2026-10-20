"""Unit tests for app.lecture.structuring: clarification tagging, chunking,
sequential chunk-by-chunk note taking with a fake LLM that returns the kind
of messy markdown small local models really produce, and the filter that
removes commentary about the lecturer / recording (no Ollama needed).

The notes must read like a student's: topic line, plain headings, bullets,
no TOC, no summary, no timecodes, no callouts, no remarks about the lecture.
"""
import re
import threading

import pytest

from app.core.errors import Cancelled
from app.lecture import export, structuring
from app.lecture.structuring import (
    DEFAULT_LLM_OPTIONS,
    EMPTY_NOTES_MARKDOWN,
    NO_CONTENT_NOTES_MARKDOWN,
    STRUCTURING_SYSTEM_PROMPT,
    chunk_tagged,
    strip_meta_commentary,
    structure_lecture,
    tag_clarifications,
)

try:
    from app.lecture.transcription import TranscribedSegment
except ImportError:  # pragma: no cover — transcription.py is owned by another module
    from dataclasses import dataclass

    @dataclass
    class TranscribedSegment:
        start: float
        end: float
        speaker: str
        text: str


class FakeLLM:
    """Stands in for OllamaClient.stream_generate.

    ``answers`` are consumed in call order; an Exception instance is raised
    instead of returned, a callable is called with the call's kwargs."""

    def __init__(self, answers=(), default=""):
        self.answers = list(answers)
        self.default = default
        self.calls = []

    def stream_generate(self, model, prompt, system=None, on_token=None, options=None,
                        cancel_event=None, timeout=(5.0, 600.0), **_kw):
        call = dict(model=model, prompt=prompt, system=system, options=options, cancel_event=cancel_event)
        self.calls.append(call)
        answer = self.answers.pop(0) if self.answers else self.default
        if callable(answer) and not isinstance(answer, Exception):
            answer = answer(call)
        if isinstance(answer, Exception):
            raise answer
        if on_token is not None:
            for word in re.findall(r"\S+\s*", answer):
                on_token(word)
        return answer


def seg(start, text, speaker="L", dur=8.0):
    return TranscribedSegment(float(start), float(start) + dur, speaker, text)


def lecture(n, step=30.0, start=0.0, clarify=()):
    """``n`` lecturer segments ``step`` seconds apart, tagged; indices in
    ``clarify`` are marked as answers to audience questions."""
    return [(seg(start + i * step, f"Фраза номер {i} о пределах и непрерывности функций.", dur=step - 2),
             i in clarify) for i in range(n)]


def split_in(monkeypatch, *bounds):
    """Force chunk boundaries at the given segment indices."""
    def chunker(tagged, max_chars=0):
        cuts = [0, *bounds, len(tagged)]
        return [list(tagged[a:b]) for a, b in zip(cuts, cuts[1:])]
    monkeypatch.setattr(structuring, "chunk_tagged", chunker)


_TIMECODE_RE = re.compile(r"\d{1,2}:\d{2}")


def student_check(md):
    """What a student's notes never contain; returns the parsed blocks."""
    for bad in ("```", "$", "\\frac", "\\lim", "<think>", "Оглавление", "Краткое содержание",
                "Уточнение по ходу лекции", "[УТОЧНЕНИЕ]", "|", ">", "Часть 1"):
        assert bad not in md, bad
    assert not _TIMECODE_RE.search(md), md
    assert not re.search(r"(?i)лектор\w*\s+(?:говорит|отмечает|переходит|подч[её]ркивает)", md), md
    for line in md.splitlines():
        assert not line.startswith("# "), line  # no stray level-1 titles
        if line.startswith("#"):
            assert re.match(r"^#{2,3} \S", line), line
            assert not re.match(r"^#{2,3} \d+[.)]", line), line  # headings are not numbered
    blocks = export.parse_markdown(md)
    assert {b.kind for b in blocks} <= {"h2", "h3", "p", "ul", "ol"}
    return blocks


def headings(md):
    return [line[3:] for line in md.splitlines() if line.startswith("## ")]


# --------------------------------------------------------------------------
# tag_clarifications
# --------------------------------------------------------------------------

def test_tag_clarifications_marks_answers_after_audience():
    segments = [
        seg(0, "a", "L"), seg(10, "вопрос", "S1"), seg(20, "ответ", "L"), seg(30, "дальше", "L"),
        seg(40, "q1", "S2"), seg(50, "q2", "S1"), seg(60, "ответ 2", "L"),
    ]
    tagged = tag_clarifications(segments, "L")
    assert [(s.text, c) for s, c in tagged] == [
        ("a", False), ("ответ", True), ("дальше", False), ("ответ 2", True)
    ]


def test_tag_clarifications_edge_cases():
    assert tag_clarifications([], "L") == []
    assert tag_clarifications([seg(0, "x", "S")], "L") == []
    # Nobody is the lecturer ("" when diarization found no speech) → nothing kept.
    assert tag_clarifications([seg(0, "x", "L")], "") == []
    only = tag_clarifications([seg(0, "x", "L"), seg(5, "y", "L")], "L")
    assert [c for _s, c in only] == [False, False]


# --------------------------------------------------------------------------
# chunk_tagged
# --------------------------------------------------------------------------

def _line_len(item):
    return len(structuring._format_line(*item)) + 1


def test_chunk_tagged_empty_and_single():
    assert chunk_tagged([]) == []
    items = lecture(3)
    assert chunk_tagged(items) == [items]


@pytest.mark.parametrize("max_chars", [200, 500, 1000, 6000])
def test_chunk_tagged_respects_limit_and_preserves_order(max_chars):
    items = lecture(97, clarify={3, 50})
    chunks = chunk_tagged(items, max_chars=max_chars)
    assert [it for chunk in chunks for it in chunk] == items  # nothing lost, reordered or split
    for chunk in chunks:
        assert chunk and sum(_line_len(it) for it in chunk) <= max_chars
    total = sum(_line_len(it) for it in items)
    assert len(chunks) >= -(-total // max_chars)


def test_chunk_tagged_is_balanced_without_tiny_tail():
    items = lecture(41)
    size = sum(_line_len(it) for it in items)
    chunks = chunk_tagged(items, max_chars=size // 3)
    sizes = [sum(_line_len(it) for it in c) for c in chunks]
    assert len(chunks) >= 3
    assert min(sizes) >= 0.5 * max(sizes), sizes


def test_chunk_tagged_oversized_line_gets_its_own_chunk():
    items = lecture(4)
    huge = (seg(200, "очень длинная фраза " * 50), False)
    items.insert(2, huge)
    chunks = chunk_tagged(items, max_chars=300)
    assert [huge] in chunks
    assert [it for c in chunks for it in c] == items


def test_prompt_lines_carry_words_only():
    # No timecodes and no markers: whatever the model sees, it echoes.
    item = (seg(65, "Ответ  на\nвопрос"), True)
    assert structuring._format_line(*item) == "Ответ на вопрос"
    assert structuring._format_line(seg(3725, "x"), False) == "x"


# --------------------------------------------------------------------------
# structure_lecture: assembly
# --------------------------------------------------------------------------

CLEAN_ANSWER = """Тема: Пределы функций

## Понятие предела
- **Предел** функции — число, к которому стремятся значения f(x) при x→a.
- Пример: 1/x → 0 при x → ∞.
- Предел может существовать, даже если f(a) не определена.

## Свойства пределов
1) Предел суммы равен сумме пределов.
2) Предел произведения равен произведению пределов.
"""


def test_structure_lecture_writes_student_notes():
    llm = FakeLLM([CLEAN_ANSWER])
    md = structure_lecture(lecture(6, clarify={1}), llm, "test-model")
    assert md.startswith("**Тема:** Пределы функций\n\n## Понятие предела\n\n- **Предел** функции")
    assert headings(md) == ["Понятие предела", "Свойства пределов"]
    assert "1. Предел суммы равен сумме пределов.\n2. Предел произведения" in md
    assert len(llm.calls) == 1  # one call per chunk, no extra summary call
    blocks = student_check(md)
    assert [b.kind for b in blocks] == ["p", "h2", "ul", "h2", "ol"]


def test_structure_lecture_passes_prompt_system_options_and_cancel_event():
    cancel = threading.Event()
    llm = FakeLLM([CLEAN_ANSWER])
    structure_lecture(lecture(4, clarify={2}), llm, "my-model", cancel_event=cancel, title_hint="Матанализ")
    first = llm.calls[0]
    assert first["model"] == "my-model"
    assert first["system"] == STRUCTURING_SYSTEM_PROMPT
    assert first["options"] == DEFAULT_LLM_OPTIONS and first["options"]["num_ctx"] >= 8192
    assert first["cancel_event"] is cancel
    prompt = first["prompt"]
    assert "Матанализ" in prompt
    assert "Фраза номер 2 о пределах" in prompt and "Фраза номер 1 о пределах" in prompt
    assert not _TIMECODE_RE.search(prompt) and "УТОЧНЕНИЕ" not in prompt
    # The rules are in the user prompt too: some Ollama templates drop the system role.
    assert "Запрещено" in prompt and "лектор говорит" in prompt and "оглавление" in prompt
    assert "Тема: …" in prompt  # the first call asks for the topic line

    custom = {"num_ctx": 16384, "temperature": 0.0, "num_predict": 1000}
    llm2 = FakeLLM([CLEAN_ANSWER])
    structure_lecture(lecture(4), llm2, "m", options=custom)
    assert llm2.calls[0]["options"] == custom
    assert custom == {"num_ctx": 16384, "temperature": 0.0, "num_predict": 1000}  # not mutated


def test_chunk_chars_kwarg_controls_chunk_size():
    items = lecture(60, step=20.0)
    llm = FakeLLM(default="- пункт")
    structure_lecture(items, llm, "m", chunk_chars=900)
    assert len(llm.calls) == len(chunk_tagged(items, max_chars=900)) >= 3
    llm = FakeLLM(default="- пункт")
    structure_lecture(items, llm, "m")
    assert len(llm.calls) == 1  # default chunk size fits the whole short lecture


def test_chunks_are_sequential_and_continue_the_notes(monkeypatch):
    items = lecture(30)
    split_in(monkeypatch, 10, 20)
    answers = [
        "Тема: Пределы\n\n## Введение\n- Пункт а.\n\n## Производная\n- Пункт б.\n- Пункт в.\n",
        # Continues the same topic: repeats the heading and echoes the last line.
        "## Производная\n- Пункт в.\n- Пункт г.\n\n## Интеграл\n- Пункт д.\n",
        # Continues without a heading at all.
        "- Пункт е.\n\n## Ряды\n- Пункт ж.\n",
    ]
    llm = FakeLLM(answers)
    md = structure_lecture(items, llm, "m")
    assert len(llm.calls) == 3
    assert headings(md) == ["Введение", "Производная", "Интеграл", "Ряды"]
    assert "- Пункт б.\n- Пункт в.\n- Пункт г.\n\n## Интеграл\n\n- Пункт д.\n- Пункт е.\n\n## Ряды" in md
    assert md.count("Пункт в.") == 1  # the echoed tail line is not repeated

    second, third = llm.calls[1]["prompt"], llm.calls[2]["prompt"]
    # Each call sees the end of the notes written so far, and only its own transcript.
    assert "## Производная\n- Пункт б.\n- Пункт в." in second
    assert "## Интеграл\n- Пункт д." in third and "Пункт г." not in third.split("---")[-3]
    assert "Фраза номер 12" in second and "Фраза номер 5 " not in second and "Фраза номер 25" not in second
    assert "Тема лекции: Пределы" in second and "Тема: …" not in second
    student_check(md)


def test_notes_tail_is_bounded():
    body = ["- " + ("длинный пункт " * 10).strip() + f" {i}" for i in range(20)]
    tail = structuring._notes_tail([structuring._Section("Тема", ["\n".join(body)])], limit=300)
    assert tail.startswith("## Тема\n") and tail.endswith(" 19")
    assert len(tail) <= 300 + len("## Тема\n") + 200
    assert structuring._notes_tail([]) == ""


# --------------------------------------------------------------------------
# structure_lecture: messy LLM output
# --------------------------------------------------------------------------

MESSY_ANSWER = r"""<think>Нужно составить конспект.</think>Конечно! Вот структурированный конспект фрагмента:

```markdown
# Конспект лекции

**Тема:** Пределы функций 📚

## Оглавление
1. Понятие предела [00:00–01:30]
2. Свойства пределов [01:30–02:58]

## Краткое содержание
Лекция посвящена пределам.

## 1. Понятие предела [00:00–01:30]
- **Предел** функции: $\lim_{x \to a} f(x) = L$.
- Лектор отмечает, что предел может не существовать.
- [00:45] Дробь $\frac{1}{x}$ и степень $x^{2}$, а также $$\sqrt{x+1}$$ (01:10).
> Уточнение по ходу лекции: значение в точке может быть не определено.

| Термин | Определение |
|---|---|
| Окрестность | интервал вокруг точки |

В данном фрагменте речь неразборчива.

## Раздел 2: Свойства пределов (01:30 - 02:58)
* Предел суммы равен сумме пределов \(\alpha + \beta\).
* Далее лектор переходит к непрерывности.
> Цитата лектора без пометки.

## Выводы
- Пределы — основа анализа.
```

Надеюсь, это поможет! Если нужно, могу дополнить."""


def test_messy_output_is_normalized():
    md = structure_lecture(lecture(6, clarify={1}), FakeLLM([MESSY_ANSWER]), "m")
    assert md.startswith("**Тема:** Пределы функций\n\n")
    assert headings(md) == ["Понятие предела", "Свойства пределов"]
    for gone in ("Конечно", "Надеюсь", "Конспект лекции", "Нужно составить", "Лекция посвящена",
                 "неразборчива", "переходит к непрерывности", "Термин", "основа анализа", "📚"):
        assert gone not in md, gone
    assert "- **Предел** функции: lim x → a f(x) = L." in md
    assert "- Предел может не существовать." in md  # «лектор отмечает, что» rewritten away
    assert "- Дробь 1/x и степень x^2, а также √(x+1)." in md
    assert "- Значение в точке может быть не определено." in md  # callout → ordinary line
    assert "- Окрестность — интервал вокруг точки" in md  # table row → plain line
    assert "- Цитата лектора без пометки." in md
    assert "α + β" in md
    student_check(md)


@pytest.mark.parametrize("raw, plain", [
    (r"$\frac{a+b}{2}$", "(a+b)/2"),
    (r"$x^{n+1}$ и $a_{ij}$", "x^(n+1) и a_ij"),
    (r"$$\sum_{i=1}^{n} i$$", "Σ_(i=1)^n i"),
    (r"\(\mathbb{R}\)", "ℝ"),
    (r"\[ a \cdot b \leq c \]", "a · b ≤ c"),
    (r"$\sqrt[3]{8} = 2$", "8^(1/3) = 2"),
    (r"путь C:\Users\name", r"путь C:\Users\name"),  # stray backslashes outside math survive
    (r"скобка \alpha вне формулы", "скобка α вне формулы"),
    ("Цена 5$ и 10$", "Цена 5$ и 10$"),  # prices are not math
])
def test_strip_latex(raw, plain):
    assert structuring._strip_latex(raw) == plain


def test_prices_keep_dollar_but_latex_leftovers_do_not():
    md = structure_lecture(lecture(3), FakeLLM(["- Цена 5$ и 10$.\n- Остаток $x^2 без пары."]), "m")
    assert "- Цена 5$ и 10$." in md and "- Остаток x^2 без пары." in md


def test_tables_become_plain_lines():
    text = structuring._normalize_text("До\n| A | B |\n|:--|--:|\n| 1 | 2 |\n|3|4|\n\nПосле\n| x | y |")
    assert text.split("\n") == ["До", "- 1 — 2", "- 3 — 4", "", "После", "- x — y"]


def test_unclosed_think_and_orphan_close_tag_are_removed():
    assert structuring._normalize_text("<think>рассуждаю бесконечно").strip() == ""
    assert structuring._normalize_text("рассуждение</think>\n- пункт").strip() == "- пункт"


def test_timecodes_never_reach_the_notes():
    answer = ("## Пределы [00:00–02:58]\n- [00:10] Определение предела.\n- 01:20 — пример с 1/x.\n"
              "- Свойства (02:30) пределов.\n**Непрерывность [02:40]**\n- Функция непрерывна (02:41–02:50).")
    md = structure_lecture(lecture(6), FakeLLM([answer]), "m")
    assert headings(md) == ["Пределы"]
    assert "- Определение предела." in md and "- Пример с 1/x." in md
    assert "### Непрерывность" in md and "- Свойства пределов." in md
    student_check(md)


def test_headingless_chunk_is_kept_without_invented_heading():
    answer = ("Сегодня говорили о рядах.\n\n- Ряд сходится, если частичные суммы имеют предел.\n"
              "- Признак Даламбера.")
    md = structure_lecture(lecture(5, start=600.0), FakeLLM([answer]), "m")
    assert headings(md) == []
    assert md == ("Сегодня говорили о рядах.\n\n- Ряд сходится, если частичные суммы имеют предел.\n"
                  "- Признак Даламбера.\n")


def test_bold_and_numbered_pseudo_headings_become_plain_headings():
    answer = "**1. Числовые ряды**\n- Определение ряда.\n\n**2. Признаки сходимости**\n- Признак сравнения."
    md = structure_lecture(lecture(6), FakeLLM([answer]), "m")
    assert headings(md) == ["Числовые ряды", "Признаки сходимости"]
    student_check(md)


def test_level1_sections_and_subheadings():
    answer = "# Пределы [00:00–01:00]\n- a\n### Пример\n- b\n\n# Производные [01:00–02:58]\n#### Детали\n- c\n"
    md = structure_lecture(lecture(6), FakeLLM([answer]), "m")
    assert headings(md) == ["Пределы", "Производные"]
    assert "### Пример" in md and "### Детали" in md
    student_check(md)


def test_lone_level1_title_on_top_is_the_topic():
    answer = "# Производная функции\n\n## Определение\n- Предел отношения приращений.\n"
    md = structure_lecture(lecture(6), FakeLLM([answer]), "m")
    assert md.startswith("**Тема:** Производная функции\n\n## Определение\n")


@pytest.mark.parametrize("first_line, topic", [
    ("Тема лекции: Древняя Русь", "Древняя Русь"),
    ("**Тема:** Древняя Русь.", "Древняя Русь"),
    ("## Тема: Древняя Русь", "Древняя Русь"),
    ("# Конспект лекции: Древняя Русь", "Древняя Русь"),
    ("# Конспект лекции по истории", ""),
    ("Тема: Лектор говорит неразборчиво", ""),
])
def test_topic_line_variants(first_line, topic):
    md = structure_lecture(lecture(3), FakeLLM([f"{first_line}\n\n## Варяги\n- Призвание варягов."]), "m")
    if topic:
        assert md.startswith(f"**Тема:** {topic}\n\n## Варяги")
    else:
        assert "Тема" not in md and md.startswith("## Варяги")


def test_title_hint_is_the_fallback_topic():
    md = structure_lecture(lecture(3), FakeLLM(["## Варяги\n- Призвание варягов."]), "m",
                           title_hint="История России")
    assert md.startswith("**Тема:** История России\n\n## Варяги")
    md = structure_lecture(lecture(3), FakeLLM(["## Варяги\n- Призвание варягов."]), "m", title_hint="rec_01")
    assert md.startswith("## Варяги")


def test_later_topic_line_becomes_a_heading(monkeypatch):
    split_in(monkeypatch, 3)
    answers = ["Тема: Древняя Русь\n\n## Варяги\n- Призвание.", "Тема: Крещение Руси\n- 988 год."]
    md = structure_lecture(lecture(6), FakeLLM(answers), "m")
    assert md.startswith("**Тема:** Древняя Русь\n\n## Варяги")
    assert headings(md) == ["Варяги", "Крещение Руси"] and "## Крещение Руси\n\n- 988 год." in md


def test_meta_and_generic_headings_are_dropped_but_content_kept():
    answer = ("## Вступление лектора\n- Сегодня — пределы.\n\n## Понятие предела\n- Определение.\n\n"
              "## Вопросы из аудитории\n- Предел может не существовать.\n\n## Продолжение\n- Ещё пункт.\n")
    md = structure_lecture(lecture(6), FakeLLM([answer]), "m")
    assert headings(md) == ["Понятие предела"]
    assert md.startswith("- Сегодня — пределы.\n\n## Понятие предела\n\n- Определение.\n"
                         "- Предел может не существовать.\n- Ещё пункт.")


def test_recap_sections_are_dropped_unless_they_are_all_there_is():
    answer = "## Пределы\n- Определение.\n\n## Итоги\n- Пределы важны.\n\n## Ключевые моменты\n- Повтор.\n"
    md = structure_lecture(lecture(6), FakeLLM([answer]), "m")
    assert headings(md) == ["Пределы"] and "важны" not in md and "Повтор" not in md
    md = structure_lecture(lecture(6), FakeLLM(["## Выводы\n- Предел единственен."]), "m")
    assert md == "- Предел единственен.\n"


def test_unheaded_and_headed_tocs_are_dropped():
    answer = ("## Смысловые разделы:\n1. Пределы\n2. Непрерывность\n\n"
              "## Пределы\n- Определение предела.\n\n## Непрерывность\n- Определение непрерывности.\n")
    md = structure_lecture(lecture(6), FakeLLM([answer]), "m")
    assert "Смысловые разделы" not in md and headings(md) == ["Пределы", "Непрерывность"]

    answer = ("План:\n- Пределы [00:00–01:28]\n- Непрерывность [01:30–02:58]\n\n"
              "## Пределы\n- Определение предела.\n\n## Непрерывность\n- Определение непрерывности.\n")
    md = structure_lecture(lecture(6), FakeLLM([answer]), "m")
    assert "План" not in md and md.count("Пределы") == 1


def test_repeated_lines_from_looping_model_are_deduplicated():
    answer = "## Пределы\n" + "- Предел — это число.\n" * 6 + "- Другое.\n"
    md = structure_lecture(lecture(6), FakeLLM([answer]), "m")
    assert md.count("Предел — это число.") == 1 and "- Другое." in md


def test_looping_answer_cut_by_num_predict_loses_its_fragment():
    loop = "".join(f"## Раздел {i}: Вопросы\nВопросы, которые могут возникнуть при изучении анализа.\n\n"
                   for i in range(3, 9))
    answer = ("## Раздел 1: Пределы\n- Определение предела.\n\n## Раздел 2: Непрерывность\n"
              "- Определение непрерывности.\n\n" + loop + "## Раздел 9: Вопросы\nВопросы, которые могут возникнуть при изуч")
    md = structure_lecture(lecture(6), FakeLLM([answer]), "m")
    assert md.count("Вопросы, которые могут") == 1
    assert not md.rstrip().endswith("изуч")
    assert headings(md) == ["Пределы", "Непрерывность", "Вопросы"]


def test_copied_format_example_does_not_leak():
    example = "\n".join(structuring._EXAMPLE_LINES)
    md = structure_lecture(lecture(3), FakeLLM([example + "\n\n## Пределы\n- Определение."]), "m")
    assert "Фотосинтез" not in md and "хлоропластах" not in md and headings(md) == ["Пределы"]


def test_instruction_echo_is_removed():
    answer = ("Конспект уже начат. Вот его последние строки (не повторяй их):\n---\n"
              "Расшифровка следующей части лекции:\n## Пределы\n- Определение.\n"
              "Напиши конспект этой части. Только конспект, без пояснений.")
    md = structure_lecture(lecture(3), FakeLLM([answer]), "m")
    assert md == "## Пределы\n\n- Определение.\n"


# --------------------------------------------------------------------------
# meta-commentary filter
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Лектор говорит невнятно.",
    "В расшифровке неразборчиво.",
    "Далее лектор переходит к свойствам пределов.",
    "В данном фрагменте рассматривается понятие предела.",
    "Фраза обрывается на полуслове.",
    "Возможно, имелось в виду «политогенез».",
    "Речь идёт о призвании варягов.",
    "Фрагмент обрывается.",
    "Эта лекция посвящена истории Руси.",
    "Смысл этой части неясен.",
    "Приветствие студентов.",
    "Студент спрашивает о зачёте.",
    "Далее речь неразборчива.",
    "(неразборчиво)",
    "Примечание: в записи много шума.",
    "Затем преподаватель возвращается к теме пределов.",
    "Конец фрагмента.",
])
def test_meta_commentary_is_dropped(text):
    assert strip_meta_commentary(text) == ""


@pytest.mark.parametrize("text, expected", [
    ("Лектор отмечает, что предел может не существовать.", "Предел может не существовать."),
    ("Также лектор подчёркивает, что **предел** единственен.", "**Предел** единственен."),
    ("По словам лектора, дата условна.", "Дата условна."),
    ("Отмечается, что дата 862 г. условна.", "Дата 862 г. условна."),
    ("Упоминание о том, что неравенство углубляется.", "Неравенство углубляется."),
    ("На вопрос студента лектор ответил, что это будет на экзамене.", "Это будет на экзамене."),
    ("Упоминается Борис Рыбаков, выдающийся археолог.", "Борис Рыбаков, выдающийся археолог."),
    ("Рюрик — основатель династии. Лектор шутит про варягов.", "Рюрик — основатель династии."),
    ("Политогенез (возможно, ошибка распознавания) — процесс формирования власти.",
     "Политогенез — процесс формирования власти."),
    ("Дата 862 г. (неразборчиво) условна.", "Дата 862 г. условна."),
    ("Ответ лектора: уточнение не требуется.", "Уточнение не требуется."),
    ("Пояснение преподавателя на вопрос о дате — дата условна.", "Дата условна."),
    ("Вопрос из аудитории: «Что?» Ответ лектора: дата условна.", "Дата условна."),
])
def test_meta_commentary_is_rewritten_to_content(text, expected):
    assert strip_meta_commentary(text) == expected


@pytest.mark.parametrize("text", [
    # A lecture *about* speech recognition keeps its content.
    "Распознавание речи — это преобразование аудиосигнала в текст.",
    "Ошибки распознавания возникают из-за шума и акцента.",
    "Невнятная речь снижает точность распознавания.",
    "Шум в записи ухудшает качество распознавания.",
    # Ordinary content that merely contains the trigger words.
    "В этой части света климат мягче.",
    "В данном фрагменте кода есть ошибка.",
    "Этот отрывок из летописи описывает призвание варягов.",
    "Организационные моменты: зачёт 25 декабря.",
    "Хорошо известно, что x^2 ≥ 0.",
    "Рюрик (возможно, легендарный персонаж) — основатель династии.",
    "**Лектор** — человек, читающий лекцию.",
    "Запись содержит набор полей.",
    "На следующей лекции — правила дифференцирования.",
    "Вопрос из зала: можно ли брать предел частного? Можно, если знаменатель не стремится к 0.",
])
def test_lecture_content_is_kept(text):
    assert strip_meta_commentary(text) == text


def test_meta_filter_applies_to_whole_notes():
    answer = ("## Пределы\n- Лектор начинает лекцию с приветствия.\n- **Предел** — число L.\n"
              "- В этом месте запись неразборчива.\n- Лектор подчёркивает, что предел единственен.\n"
              "Далее лектор переходит к примерам.\n")
    md = structure_lecture(lecture(6), FakeLLM([answer]), "m")
    assert md == "## Пределы\n\n- **Предел** — число L.\n- Предел единственен.\n"


# --------------------------------------------------------------------------
# structure_lecture: failures, empties, progress, cancel
# --------------------------------------------------------------------------

def test_empty_input_returns_note_without_llm_calls():
    progress = []
    llm = FakeLLM()
    assert structure_lecture([], llm, "m", on_progress=progress.append) == EMPTY_NOTES_MARKDOWN
    assert structure_lecture([(seg(0, "  "), False)], llm, "m") == EMPTY_NOTES_MARKDOWN
    assert llm.calls == [] and progress == [1.0]
    assert "не распознана речь лектора" in EMPTY_NOTES_MARKDOWN
    assert [b.kind for b in export.parse_markdown(EMPTY_NOTES_MARKDOWN)] == ["h2", "p"]


def test_no_usable_answer_gives_a_short_note_not_a_transcript_dump():
    llm = FakeLLM(["", "   "])
    md = structure_lecture(lecture(3, clarify={1}), llm, "m")
    assert len(llm.calls) == 2  # one retry on an empty answer
    assert md == NO_CONTENT_NOTES_MARKDOWN and "Фраза номер" not in md
    md = structure_lecture(lecture(3), FakeLLM(["Лектор говорит неразборчиво. Фраза обрывается."]), "m")
    assert md == NO_CONTENT_NOTES_MARKDOWN


def test_chatter_only_recording_gives_the_stub_without_calling_the_model():
    # Real case: a 5-minute recording with "Да. Сейчас секунду. Оба. Пока. Что?" made a
    # model invent a topic and a lecturer's "answer".
    chatter = ["Да.", "Сейчас секунду.", "Оба", "Пока", "Что?", "Ладно, давайте."]
    tagged = [(seg(i * 20, text), i == 4) for i, text in enumerate(chatter)]
    assert not structuring.has_lecture_material(tagged)
    llm, progress = FakeLLM(default="Тема: Проверка понимания\n- Вопрос из аудитории: «Что?»"), []
    md = structure_lecture(tagged, llm, "m", on_progress=progress.append)
    assert md == NO_CONTENT_NOTES_MARKDOWN and llm.calls == [] and progress == [1.0]
    assert "Модель" not in md
    assert structuring.has_lecture_material(lecture(3))  # 3 short content sentences are enough


def test_garbage_chunk_between_good_ones_is_skipped(monkeypatch):
    split_in(monkeypatch, 3, 6)
    answers = ["## Пределы\n- Определение.", "В данном фрагменте речь неразборчива.", "- Пример с 1/x."]
    md = structure_lecture(lecture(9), FakeLLM(answers), "m")
    assert md == "## Пределы\n\n- Определение.\n- Пример с 1/x.\n"


def test_chunk_error_is_retried_once_then_propagates():
    llm = FakeLLM([RuntimeError("timeout"), CLEAN_ANSWER])
    md = structure_lecture(lecture(6), llm, "m")
    assert len(llm.calls) == 2 and "Понятие предела" in md

    llm = FakeLLM([RuntimeError("down"), RuntimeError("still down")])
    with pytest.raises(RuntimeError, match="still down"):
        structure_lecture(lecture(6), llm, "m")


def test_progress_is_monotonic_and_reaches_one(monkeypatch):
    split_in(monkeypatch, 10, 20)
    progress = []
    structure_lecture(lecture(30), FakeLLM([CLEAN_ANSWER] * 3), "m", on_progress=progress.append)
    assert progress[-1] == 1.0
    assert all(0.0 <= p <= 1.0 for p in progress)
    assert progress == sorted(progress)
    for done in (1 / 3, 2 / 3):  # reported after every chunk
        assert any(abs(p - done) < 1e-9 for p in progress), done


def test_cancel_before_start_and_during_call(monkeypatch):
    cancel = threading.Event()
    cancel.set()
    llm = FakeLLM([CLEAN_ANSWER])
    with pytest.raises(Cancelled):
        structure_lecture(lecture(6), llm, "m", cancel_event=cancel)
    assert llm.calls == []

    cancel = threading.Event()
    split_in(monkeypatch, 10)

    def cancelling(call):
        cancel.set()  # user pressed "Отменить" while tokens were streaming
        return "## Обрывок"

    llm = FakeLLM([cancelling, CLEAN_ANSWER])
    with pytest.raises(Cancelled):
        structure_lecture(lecture(20), llm, "m", cancel_event=cancel)
    assert len(llm.calls) == 1

    cancel = threading.Event()

    def failing(call):
        cancel.set()
        return RuntimeError("connection closed by cancel")

    with pytest.raises(Cancelled):
        structure_lecture(lecture(20), FakeLLM([failing]), "m", cancel_event=cancel)


def test_assembled_notes_export_to_plain_docx(tmp_path):
    from docx import Document

    md = structure_lecture(lecture(6, clarify={1}), FakeLLM([MESSY_ANSWER]), "m")
    out = tmp_path / "notes.docx"
    export.export_notes_docx(md, out, "Конспект лекции — 06.10.2026 10:00")
    paras = [p for p in Document(str(out)).paragraphs if p.text.strip()]
    styles = [p.style.name for p in paras]
    assert styles.count("Heading 2") == 2
    assert "List Bullet" in styles and "Intense Quote" not in styles and "Quote" not in styles
    assert paras[1].text == "Тема: Пределы функций" and paras[1].runs[0].bold
    assert not any("**" in p.text or p.text.startswith(("- ", "> ")) for p in paras)
