"""Turns the lecturer-only transcript into lecture notes («конспект») that
read like a student's own notes, not like generated text.

What a student writes, and therefore what this module produces: a topic
line, a short plain heading whenever the lecturer moves to a new topic,
dash bullets and short sentences with the content itself, definitions with
the term in bold, formulas as plain text, the examples and conclusions the
lecturer gave — strictly in lecture order. What a student never writes and
the user explicitly rejected in earlier versions: a table of contents, a
summary block, timecodes (they stay in the transcript only), numbered
sections, tables, coloured callout boxes ("Уточнение по ходу лекции") and,
above all, remarks *about* the lecture or the recording («лектор отмечает,
что…», «в данном фрагменте…», «речь неразборчива»).

The transcript still has to be cut into ~6000-character chunks: one prompt
with a whole lecture is silently truncated by Ollama's context window. To
keep the notes one continuous text instead of a stack of per-chunk
mini-documents, the chunks are processed sequentially and every call sees
the tail of the notes written so far (last heading + last lines), so the
model continues where it stopped. A heading repeated across a chunk
boundary is merged in code.

Small local models rarely follow format instructions exactly (they add a
TOC or a preamble, wrap the answer in code fences, emit LaTeX or tables,
leak <think> blocks, number headings, narrate what the lecturer does), so
every answer goes through a forgiving normalizer and a meta-commentary
filter that map it onto the markdown subset export.py renders: ``##``/
``###`` headings, ``- `` bullets, ``1. `` items, paragraphs and inline
``**bold**``/``*italic*``. The filter rewrites «лектор отмечает, что X» to
«X» and drops sentences that only talk about the lecturer, the recording or
the transcript; it is anchored on deictic/narrative wording so a lecture
*about* speech recognition keeps its content.

The rules are repeated in the user prompt because some models' Ollama
templates (YandexGPT) silently drop the system role.
"""
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from app.core.errors import Cancelled
from app.lecture.transcription import TranscribedSegment

try:  # models_config is being reworked (per-tier options); never depend on one constant
    from app.core.models_config import LECTURE_LLM_OPTIONS as _CONFIG_OPTIONS
except ImportError:  # pragma: no cover — depends on the other module's version
    _CONFIG_OPTIONS = None

logger = logging.getLogger(__name__)

DEFAULT_LLM_OPTIONS = dict(_CONFIG_OPTIONS or {"num_ctx": 8192, "temperature": 0.2, "num_predict": 2048})
DEFAULT_CHUNK_CHARS = 6000
PHOTO_PROMPT_MAX_CHARS = 2500  # photo text per chunk prompt (~800 tokens of the 8192 context)
PHOTO_SPLIT_MIN_CHARS = 800    # a photo moment splits a chunk only if both parts have this much speech (~3 min)
PHOTOS_SECTION_TITLE = "Фото с занятия"
_IMAGE_LINE_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$")


@dataclass
class NotesPhoto:
    """A photo for the notes: where its file is (relative to notes.md),
    its caption, when it was taken and what the vision model read on it."""
    src: str                         # e.g. "photos/001.jpg"
    caption: str                     # e.g. «Фото 1»
    offset_s: Optional[float] = None
    text: str = ""
TAIL_CHARS = 600  # how much of the notes written so far each next call sees

STRUCTURING_SYSTEM_PROMPT = (
    "Ты — студент, который аккуратно конспектирует лекцию на русском языке. Ты записываешь только "
    "сам материал лекции, по порядку, своими словами, без комментариев о лекторе, записи или "
    "расшифровке и без оглавлений, итогов и таймкодов."
)

_RULES = """Как пишется конспект:
- Записывай сам материал лекции своими словами, коротко и строго по порядку, как его излагает лектор.
- Когда лектор переходит к новой теме, поставь короткий заголовок строкой «## Название темы» (2–5 слов, без номера). Пока тема прежняя, новый заголовок не ставь.
- Под заголовком — пункты, каждый с новой строки и с «- »: определения, факты, даты, имена, формулы, примеры, выводы лектора. Можно короткими предложениями.
- Определение пиши так: «- **Термин** — определение». Жирным выделяй только термины.
- Если лектор что-то перечисляет, пиши «1) …», «2) …».
- Формулы пиши обычным текстом: x^2, a/b, √x, lim x→a f(x).
- Ответ лектора на вопрос из аудитории запиши обычным пунктом, по сути ответа.
- Обрывки фраз, оговорки и организационные реплики («так», «подождите», «тише») пропускай. Очевидные ошибки распознавания исправляй молча.

Запрещено:
- писать о лекторе, записи или расшифровке: «лектор говорит / отмечает / переходит к…», «в данном фрагменте…», «речь идёт о…», «неразборчиво», «фраза обрывается»;
- оглавление, план, краткое содержание, итоги и выводы от себя, вступление и заключение от себя;
- таблицы, цитаты «>», таймкоды, LaTeX и символ $, эмодзи, вопросы к читателю, пояснения о том, что ты делаешь."""

# The example shows the format; its lines are pre-seeded into the dedupe set
# so a model that copies it verbatim doesn't leak them into the notes.
_EXAMPLE_LINES = (
    "## Фотосинтез",
    "- **Фотосинтез** — образование органических веществ из CO2 и воды на свету.",
    "- Идёт в хлоропластах, нужен хлорофилл.",
    "- Итог: 6CO2 + 6H2O → C6H12O6 + 6O2.",
)
_EXAMPLE = "Пример оформления (на другую тему):\n" + "\n".join(_EXAMPLE_LINES)

EMPTY_NOTES_MARKDOWN = (
    "## Конспект не составлен\n"
    "\n"
    "В записи не распознана речь лектора, поэтому составить конспект не из чего. "
    "Проверьте расшифровку: возможно, запись слишком тихая или речь не была распознана.\n"
)

NO_CONTENT_NOTES_MARKDOWN = (
    "## Конспект не составлен\n"
    "\n"
    "В речи лектора не нашлось учебного материала для конспекта — только отдельные реплики. "
    "Полный текст записи — в расшифровке.\n"
)

# A few stray words ("Да. Сейчас секунду. Что?") are not lecture material: a
# model asked to take notes on them invents a topic and an "answer", which a
# student would never write. Below this many content words the model isn't
# called at all.
MIN_CONTENT_WORDS = 12
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]{3,}")
_FILLER_WORDS = frozenset(
    "да нет ага угу ну так вот ладно хорошо окей сейчас секунду секунда секундочку минуту минутку "
    "подождите подожди пока что это эта эти этот как там тут здесь все всё ещё еще уже тоже может "
    "тогда вообще значит короче типа давайте давай ребят ребята понятно понял поняли спасибо тише "
    "внимание слышно видно нормально правильно точно конечно просто кто где чего чем тем том его она "
    "они оно мне меня вас нас вам нам ваш наш так-так знаю знаете смотрите слушайте можно нужно надо "
    "были был была было есть или если только очень потом теперь сегодня".split()
)


def has_lecture_material(tagged: Sequence["Tagged"]) -> bool:
    """True when the lecturer said enough content words to take notes on.

    Fillers and short acknowledgements don't count, so a recording of
    chatter ("Ладно. Сейчас секунду.") yields the "nothing to note" stub
    without loading a model."""
    count = 0
    for seg, _clar in tagged:
        for word in _WORD_RE.findall(seg.text or ""):
            if word.lower() not in _FILLER_WORDS:
                count += 1
                if count >= MIN_CONTENT_WORDS:
                    return True
    return False


_MAX_TITLE_CHARS = 90
_MAX_TOPIC_CHARS = 100

Tagged = Tuple[TranscribedSegment, bool]


# --------------------------------------------------------------------------
# Input side: tagging and chunking
# --------------------------------------------------------------------------

def tag_clarifications(segments: Sequence[TranscribedSegment], lecturer: str) -> List[Tagged]:
    """Keep only lecturer segments; a lecturer segment that immediately
    follows a non-lecturer segment is treated as an answer to a question
    from the audience (the flag is kept for callers; the notes write such
    answers as ordinary lines)."""
    tagged = []
    prev_was_audience = False
    for seg in segments:
        is_lecturer = seg.speaker == lecturer
        is_clarification = is_lecturer and prev_was_audience
        if is_lecturer:
            tagged.append((seg, is_clarification))
        prev_was_audience = not is_lecturer
    return tagged


def _format_line(seg: TranscribedSegment, is_clarification: bool = False) -> str:
    """One transcript line of the prompt: the words only. Timecodes and
    markers are left out on purpose — whatever the model sees, it echoes,
    and neither belongs in a student's notes."""
    return " ".join((seg.text or "").split())


def chunk_tagged(tagged: Sequence[Tagged], max_chars: int = DEFAULT_CHUNK_CHARS) -> List[List[Tagged]]:
    """Split tagged segments into consecutive chunks whose prompt text (one
    line per segment) stays within ``max_chars``.

    Chunks are balanced (similar sizes) instead of greedily filled, so the
    last one is not a tiny leftover the model would pad with filler. A
    single line longer than ``max_chars`` gets a chunk of its own; segments
    are never split or reordered.
    """
    items = list(tagged)
    if not items:
        return []
    max_chars = max(1, int(max_chars))
    sizes = [len(_format_line(seg, clar)) + 1 for seg, clar in items]
    total = sum(sizes)
    # 10 % headroom: line granularity must not push a chunk over the limit
    # and spill a tiny extra chunk.
    n_chunks = max(1, math.ceil(total / (max_chars * 0.9)))
    target = total / n_chunks

    chunks: List[List[Tagged]] = []
    sizes_out: List[int] = []
    cur: List[Tagged] = []
    cur_size = 0
    for item, size in zip(items, sizes):
        if cur and (cur_size + size > max_chars or cur_size + size / 2 > target):
            chunks.append(cur)
            sizes_out.append(cur_size)
            cur, cur_size = [], 0
        cur.append(item)
        cur_size += size
    if cur:
        if chunks and cur_size < target * 0.25 and sizes_out[-1] + cur_size <= max_chars:
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)
    return chunks


def split_at_moments(chunks: Sequence[Sequence[Tagged]], moments: Sequence[float],
                     min_chars: int = PHOTO_SPLIT_MIN_CHARS) -> List[List[Tagged]]:
    """Chunks cut again at the given moments (seconds), so a photo can sit
    between the notes of what was said before and after it was taken.

    WHY: a chunk is ~6000 chars of speech (16–20 min); a photo placed after
    its chunk's notes landed up to 20 min of material late, and in a short
    recording "по времени" looked the same as "в конце". A cut is made only
    when both parts keep ``min_chars``: tiny chunks make the model pad and
    cost a call each. Segments are never split or reordered.
    """
    out: List[List[Tagged]] = []
    marks = sorted(float(m) for m in moments if m is not None)
    for chunk in chunks:
        pieces = [list(chunk)]
        for moment in marks:
            last = pieces[-1]
            cut = next((j for j, (seg, _clar) in enumerate(last) if float(seg.start) >= moment), None)
            if not cut:  # before this piece (0) or after it (None)
                continue
            sizes = [len(_format_line(seg, clar)) + 1 for seg, clar in last]
            if sum(sizes[:cut]) >= min_chars and sum(sizes[cut:]) >= min_chars:
                pieces[-1:] = [last[:cut], last[cut:]]
        out.extend(pieces)
    return out


# --------------------------------------------------------------------------
# Meta-commentary filter
# --------------------------------------------------------------------------

_LECT = r"(?:лектор\w{0,3}|преподавател\w{0,4}|преподавательниц\w{0,2}|спикер\w{0,3}|докладчик\w{0,3})"
# Narrative verbs ("лектор <verb>…"): stems, so every tense/aspect form matches.
_VERB = (
    r"(?:говор|сказа|отмеча|отметил|подч[её]ркива|подчеркнул|переход|перешёл|перешел|рассказ|упомина|"
    r"упомянул|объясня|объяснил|привод|привёл|привел|зада|спрашива|спросил|проси|просит|попросил|обраща|"
    r"обратил|начина|начал|продолжа|продолжил|заканчива|закончил|заверша|завершил|возвраща|вернул|"
    r"напомина|напомнил|шут|пошутил|уточня|уточнил|поясня|пояснил|отвеча|ответил|затрагива|затронул|"
    r"описыва|описал|рассматрива|рассмотрел|обсужда|обсудил|предлага|предложил|акцентир|делает|сделал|"
    r"демонстрир|показыва|показал|рису|нарисовал|пишет|написал|записыва|записал|выража|высказыва|"
    r"считает|утвержда|указыва|указал|ссыла|сослал|цитир|комментир|прокомментир|замеча|заметил|ставит|"
    r"поставил|подвод|подв[её]л|резюмир|называ|назвал|интересу|проверя|провер|призыва|призвал|советует|"
    r"посоветовал|рекоменд|выделя|выделил|формулир|сформулир|сравнива|сравнил|иллюстрир|поднима|поднял|"
    r"касает|коснул|приступа|приступил|переключа|отвлека|отвл[её]к|извиня|благодар|здорова|проща|"
    r"повторя|повторил|признаёт|признает|сомнева|допуска|предполага|раскрыва|раскрыл|излага|изложил|"
    r"разбира|разобрал|переш)"
)
_DEIX = r"(?:эт|данн|настоящ|представленн|приведённ|приведенн|текущ)\w*"
_ADV = r"(?:(?:также|далее|затем|потом|кроме\s+того|при\s+этом|ещё|еще|здесь|тут|отдельно|в\s+целом)\s*,?\s+)"
# Genitive nouns after "фрагмент/отрывок/текст" that make it lecture content
# ("отрывок из летописи", "фрагмент кода"), not a remark about the recording.
_CONTENT_OF = (
    r"(?!\s+(?:из|летопис|текст|произведени|роман|поэм|книг|документ|договор|закон|код|программ|"
    r"стих|рассказ|повест|письм|сочинени|источник|днк|рнк|молекул|бел|ген|цеп|карт|граф|изображени)\w*)"
)
_SENT_START = r"^[\s\"«(*]*"
# What follows "этот фрагмент / данная лекция / эта часть" when it is a remark
# about the recording ("…рассматривается", "…лектор", end of clause), as
# opposed to content ("в этой части света", "в данном фрагменте кода").
_META_FOLLOW = (
    r"(?=\s*(?:[,.:;!?)]|$)|\s+(?:лекции|записи|расшифровки|фрагмента|занятия|мы\b|рассматр|обсужд|говор|речь|"
    r"упомина|затрагива|лектор|преподават|есть\b|нет\b|присутств|встреча|содерж|описыва|привод|излага|разбира|"
    r"перечисл|да[её]тся|посвящ|начина|заканчива|заверша|обрыва|прерыва|продолжа|касает|состоит|включа|"
    r"неразборч|невнятн|представлен|идёт|идет|шла))"
)

_META_REWRITE_RES = [
    # «Лектор (также) отмечает, что X» / «Преподаватель подчёркивает, что X» → X
    re.compile(
        _SENT_START + _ADV + r"?" + _LECT + r"\s+(?:\S+\s+){0,3}?" + _VERB + r"\w*\s*,?\s*(?:о\s+том\s*,?\s*)?"
        r"(?:что|чтобы)\s+(?P<rest>.+)$", re.IGNORECASE | re.S),
    # «По словам лектора, X» / «Как отметил лектор, X»
    re.compile(_SENT_START + r"(?:по\s+словам|по\s+мнению|со\s+слов)\s+" + _LECT + r"\s*,\s*(?P<rest>.+)$",
               re.IGNORECASE | re.S),
    re.compile(_SENT_START + r"как\s+(?:\S+\s+)?" + _VERB + r"\w*\s+" + _LECT + r"\s*,\s*(?P<rest>.+)$",
               re.IGNORECASE | re.S),
    # «Отмечается / Было отмечено / Подчёркивается, что X»
    re.compile(
        _SENT_START + _ADV + r"?(?:было\s+|был\s+|была\s+)?(?:отмечается|отмечено|подч[её]ркивается|подч[её]ркнуто|"
        r"упоминается|упомянуто|говорится|сказано|утверждается|указывается|указано|поясняется|пояснено|"
        r"объясняется|объяснено|уточняется|уточнено|констатируется|выделяется|делается\s+вывод|"
        r"сделан\s+вывод|звучит\s+мысль)\s*,?\s*(?:о\s+том\s*,?\s*)?что\s+(?P<rest>.+)$", re.IGNORECASE | re.S),
    re.compile(_SENT_START + r"(?:упоминание|утверждение|мысль|идея)\s+(?:о\s+)?том\s*,\s*что\s+(?P<rest>.+)$",
               re.IGNORECASE | re.S),
    # «На вопрос (студента) лектор ответил, что X» / «…ответил: X»
    re.compile(
        _SENT_START + r"(?:отвечая\s+на\s+вопрос\w*|на\s+вопрос\w*)\b[^,:]{0,80},?\s*(?:" + _LECT + r"\s+)?"
        r"(?:ответил|отвечает|пояснил|поясняет|уточнил|уточняет|сказал|говорит|отметил|отмечает)\w*"
        r"\s*(?:,?\s*что\s+|\s*[:—–-]\s*)(?P<rest>.+)$", re.IGNORECASE | re.S),
    # «Ответ лектора: X» / «Пояснение преподавателя на вопрос о Y — X» → X
    re.compile(
        _SENT_START + r"(?:ответ|пояснение|комментарий|уточнение)\s+" + _LECT +
        r"(?:\s+на\s+вопрос\w*[^:—–]{0,60})?\s*[:—–]\s*(?P<rest>.+)$", re.IGNORECASE | re.S),
    # «Упоминается Борис Рыбаков…» / «Приводится пример: …» → the thing itself
    re.compile(
        _SENT_START + r"(?:упоминается|упоминаются|приводится|приводятся|рассматривается|рассматриваются|"
        r"обсуждается|обсуждаются|описывается|описываются|разбирается|разбираются)\s+(?P<rest>(?!что\b)\S.+)$",
        re.IGNORECASE | re.S),
]

_META_DROP_RES = [
    # the lecturer as the narrated subject: «лектор переходит к…», «отмечает лектор»
    re.compile(r"(?<!\w)" + _LECT + r"\s+(?:\S+\s+){0,3}?" + _VERB, re.IGNORECASE),
    re.compile(r"(?<!\w)" + _VERB + r"\w*\s+" + _LECT + r"(?!\w)", re.IGNORECASE),
    # «В данном фрагменте / В расшифровке / Во фрагменте …»
    re.compile(_SENT_START + _ADV + r"?(?:в|во|из|на)\s+" + _DEIX + r"\s+(?:фрагмент\w*|отрывк\w*|отрыв\w*|"
               r"расшифровк\w*|транскрипци\w*|транскрипт\w*|аудиозапис\w*|запис\w*|лекци\w*|част\w*|текст\w*)"
               + _META_FOLLOW, re.IGNORECASE),
    re.compile(_SENT_START + _ADV + r"?(?:в|во|из)\s+(?:фрагмент\w*|отрывк\w*|расшифровк\w*|транскрипци\w*|"
               r"аудиозапис\w*)\s*(?:,|$|(?:есть|нет|встреча|присутству|име|указ|говор|сказ|отсутству|не\b|"
               r"неразборч|обрыва|повторя|содерж|упомина|рассматр|обсужд|речь))", re.IGNORECASE),
    # «Этот/Данный фрагмент…», «Эта лекция посвящена…»
    re.compile(r"(?<!\w)" + _DEIX + r"\s+(?:фрагмент\w*|отрывк\w*|отрыв\w*|расшифровк\w*|транскрипци\w*|"
               r"транскрипт\w*|аудиозапис\w*|запис[иь]\w*|лекци\w*|части\s+лекции)" + _CONTENT_OF + _META_FOLLOW,
               re.IGNORECASE),
    # «Лекция начинается с…», «Фрагмент обрывается»
    re.compile(_SENT_START + r"(?:лекция|занятие|фрагмент|отрывок|расшифровка|транскрипция|аудиозапись)\s+"
               r"(?:\S+\s+)?(?:посвящ|начина|заканчива|заверша|обрыва|прерыва|продолжа|содерж|включа|охватыва|"
               r"касается|затрагива|рассказыва|описыва|представля|состоит)\w*", re.IGNORECASE),
    re.compile(r"(?<!\w)(?:фраза|предложение|мысль|текст|речь|запись|фрагмент|расшифровка|лекция|часть|слово)\s+"
               r"(?:\S+\s+){0,2}?(?:обрывается|оборван\w*|прерывается|прерван\w*|не\s+законч\w+|не\s+дописан\w*|"
               r"утерян\w*|потерян\w*|не\s+распознан\w*)", re.IGNORECASE),
    re.compile(r"на\s+полуслове", re.IGNORECASE),
    re.compile(r"(?<!\w)(?:смысл|суть|контекст)\w*\s+(?:\S+\s+){0,2}?(?:неясе?н|не\s+ясе?н|непоняте?н|"
               r"не\s+поняте?н|сложно\s+понять|трудно\s+понять|не\s+удаётся\s+понять|не\s+удается\s+понять|"
               r"теряется|утерян)", re.IGNORECASE),
    re.compile(r"(?<!\w)(?:трудно|сложно|невозможно|нельзя)\s+(?:понять|разобрать|установить|определить)\s*,?\s*"
               r"(?:о\s+ч[её]м|что\s+именно|что\s+име|что\s+хотел|какой|какая|какое)", re.IGNORECASE),
    # recording quality — only next to words that tie it to the recording
    re.compile(r"(?<!\w)(?:неразборчив\w*|невнятн\w*|нерасслыш\w*|не\s+расслыш\w*|плохо\s+(?:слышн|различим|"
               r"разборчив)\w*|не\s+слышн\w*|не\s+удалось\s+(?:разобрать|расслышать|распознать)|нрзб)"
               r"(?=.*?(?<!\w)(?:здесь|тут|далее|дальше|затем|в\s+этом\s+месте|последующ\w*|остальн\w*|оставш\w*|"
               r"част\w*|фрагмент\w*|отрыв\w*|запис\w*|расшифровк\w*|лектор\w*|преподавател\w*|сказан\w*|"
               r"произн[её]с\w*|фраз\w*|слов[аоу]?|предложени\w*|конец|начало|окончани\w*)(?!\w))", re.IGNORECASE),
    re.compile(r"(?<!\w)(?:здесь|тут|далее|дальше|затем|в\s+этом\s+месте|последующ\w*|остальн\w*|оставш\w*|"
               r"част\w*|фрагмент\w*|отрыв\w*|запис\w*|расшифровк\w*|лектор\w*|преподавател\w*|сказан\w*|"
               r"произн[её]с\w*|фраз\w*|слов[аоу]?|предложени\w*|конец|начало|окончани\w*)(?!\w)"
               r".*?(?<!\w)(?:неразборчив\w*|невнятн\w*|нерасслыш\w*|не\s+расслыш\w*|плохо\s+(?:слышн|различим|"
               r"разборчив)\w*|не\s+слышн\w*|не\s+удалось\s+(?:разобрать|расслышать|распознать)|нрзб)",
               re.IGNORECASE),
    re.compile(_SENT_START + r"(?:неразборчиво|невнятно|не\s*расслышано|нрзб)\W*$", re.IGNORECASE),
    re.compile(_SENT_START + r"(?:в\s+|во\s+)?(?:качество\s+)?(?:аудио)?запис\w*\s+(?:\S+\s+){0,2}?(?:плох|низк|"
               r"некачеств|шум|помех|тих|зашумл|эхо|обрыв|прерыв|неразборч)", re.IGNORECASE),
    # the model's guesses about ASR errors
    re.compile(_SENT_START + r"(?:вероятно|возможно|по-видимому|видимо|скорее\s+всего|предположительно|похоже)"
               r"\s*,?\s+(?:(?:здесь|тут)\s+)?(?:имел\w*\s+в\s+виду|име\w*\s+в\s+виду|речь\s+(?:ид[её]т|шла)|"
               + _LECT + r"|это\s+(?:ошибка|оговорка|опечатка)|ошибк\w*\s+распознавани|оговорк|опечатк)",
               re.IGNORECASE),
    re.compile(r"(?:ошибк\w*|искажени\w*)\s+(?:распознавани\w*|расшифровк\w*|транскрипци\w*)"
               r"(?=.*?(?:вероятно|возможно|видимо|по-видимому|похоже|скорее))|"
               r"(?:вероятно|возможно|видимо|по-видимому|похоже|скорее\s+всего)[^.]{0,40}?(?:ошибк\w*|искажени\w*)"
               r"\s+(?:распознавани\w*|расшифровк\w*|транскрипци\w*)", re.IGNORECASE),
    # narration of the lecture flow: «Далее рассматривается…», «Затем лектор…»
    re.compile(_SENT_START + r"(?:далее|затем|потом|после\s+этого|в\s+начале|в\s+конце|в\s+заключение|"
               r"в\s+завершение|в\s+продолжение|дальше)\s*,?\s+(?:\S+\s+){0,3}?(?:" + _LECT + r"|лекци\w*|"
               r"занят\w*|речь\s+(?:ид[её]т|пойд[её]т|шла)|рассматрива\w*|обсужда\w*|говорится|переход\w*|"
               r"следует\s+переход)", re.IGNORECASE),
    re.compile(_SENT_START + r"(?:в|на)\s+(?:(?:эт|данн|сегодняшн|настоящ)\w*\s+)?(?:лекции|занятии|уроке|"
               r"семинаре)\s+(?:\S+\s+)?(?:рассматрива|обсужда|говорит|говорил|речь|упомина|затрагива|разбира|"
               r"был\w*\s+рассмотр|был\w*\s+затронут)\w*", re.IGNORECASE),
    re.compile(_SENT_START + r"речь\s+(?:ид[её]т|шла|пойд[её]т)\s+(?:о|об|про)(?!\w)", re.IGNORECASE),
    # the audience narrated
    re.compile(_SENT_START + r"(?:(?:один\s+из\s+)?(?:студент\w*|слушател\w*)|кто-то\s+из\s+(?:студентов|"
               r"слушателей|зала|аудитории)|из\s+(?:зала|аудитории))\s+(?:\S+\s+){0,2}?(?:спрашива|спросил|"
               r"зада|задал|интересу|уточня|уточнил|отвеча|ответил|смеют|смеёт|смеет|шум|перебива|прерыва)\w*",
               re.IGNORECASE),
    re.compile(_SENT_START + r"(?:прозвучал\w*|был\w*\s+задан\w*|поступил\w*)\s+(?:\S+\s+)?вопрос",
               re.IGNORECASE),
    # a labelled audience question with no substance: «Вопрос из аудитории: «Что?»»
    # (a real question with its answer is kept — students do write those down)
    re.compile(_SENT_START + r"вопрос\w*\s+(?:из\s+(?:зала|аудитории)|студент\w*|слушател\w*|аудитории)"
               r"\s*\**\s*[:—–]\s*(?:\S+\s*){0,2}$", re.IGNORECASE),
    # pure filler items
    re.compile(_SENT_START + r"(?:приветствие|обращение\s+к\s+(?:аудитории|студентам|слушателям)|"
               r"организационн\w+\s+(?:момент|вопрос)\w*|перерыв|пауза|тишина|шум\s+в\s+аудитории|"
               r"несвязн\w*\s+(?:реплик|фраз)\w*|обрывки\s+фраз\w*|отдельные\s+(?:слова|фразы|реплики)|"
               r"реплик\w*\s+(?:из\s+зала|студентов|аудитории))(?:\s[^.:]{0,40})?\.?\s*$", re.IGNORECASE),
    re.compile(_SENT_START + r"(?:конец|окончание|продолжение)\s+(?:фрагмента|лекции|записи|расшифровки|следует)",
               re.IGNORECASE),
]

# Inline remarks in brackets: «(неразборчиво)», «[возможно, имелось в виду …]».
_META_PAREN_RE = re.compile(
    r"\s*[(\[]\s*(?:неразборчиво|не\s*разборчиво|невнятно|не\s*расслышано|нрзб\.?|неясно|непонятно|inaudible|"
    r"unclear|пауза|смех|шум|\?{1,3}|…|\.{3}|sic!?|так\s+в\s+(?:записи|расшифровке|тексте|оригинале)|"
    r"далее\s+неразборчиво|фраза\s+оборвана|обрыв\s+(?:фразы|записи|речи)|"
    r"[^()\[\]]{0,60}?(?:имел\w*\s+в\s+виду|ошибк\w*\s+распознав|оговорк|опечатк|искажен|расслыш|"
    r"неразборчив|в\s+записи|в\s+расшифровке|лектор)[^()\[\]]{0,60}?)\s*[)\]]",
    re.IGNORECASE,
)
_NOTE_LABEL_RE = re.compile(r"^(?P<m>\**)\s*(?:примечание|комментарий|прим\.|note)\s*\**\s*[:—–-]\s*\**\s*",
                            re.IGNORECASE)
_DEFINITION_RE = re.compile(r"^\s*\*\*[^*]+\*\*\s*(?:—|–|-|:)|\s—\s*это\b|(?<!\w)называ(?:ется|ют)(?!\w)",
                            re.IGNORECASE)
# A sentence may end inside a closing quote/bracket: «Вопрос: «Что?» Ответ: …».
_SENTENCE_SPLIT_RE = re.compile(r"(?:(?<=[.!?…])|(?<=[.!?…][»\")]))\s+(?=[«\"(*]*[A-ZА-ЯЁ0-9])")


def _capitalize(text: str) -> str:
    m = re.match(r"^([\s\"«(*]*)([а-яё])", text)
    if m:
        return text[:m.start(2)] + m.group(2).upper() + text[m.end(2):]
    return text


def _filter_sentence(sentence: str) -> str:
    s = sentence.strip()
    if not s:
        return ""
    for _ in range(2):  # «Также отмечается, что лектор говорит, что X» needs two passes
        for rx in _META_REWRITE_RES:
            m = rx.match(s)
            if m and m.group("rest").strip():
                lead = re.match(r"^[\s\"«(*]*", s).group(0)
                s = _capitalize(lead.replace(" ", "") + m.group("rest").strip())
                break
        else:
            break
    if _DEFINITION_RE.search(s):
        return s
    probe = _strip_emphasis(s)
    for rx in _META_DROP_RES:
        if rx.search(probe):
            return ""
    return s


def strip_meta_commentary(text: str) -> str:
    """Remove remarks about the lecturer / recording / transcript from one
    line of notes, keeping the content: «Лектор отмечает, что X» → «X»,
    «В данном фрагменте речь неразборчива.» → "". Returns "" when nothing
    of substance is left."""
    text = _META_PAREN_RE.sub("", text or "")
    labelled = _NOTE_LABEL_RE.sub(lambda m: m.group("m"), text.strip())
    text = _capitalize(labelled) if labelled != text.strip() else labelled
    sentences = _SENTENCE_SPLIT_RE.split(text)
    kept = [s for s in (_filter_sentence(x) for x in sentences) if s]
    out = " ".join(kept).strip()
    if out.count("**") % 2:
        out = out.replace("**", "")
    if not re.search(r"\w", _strip_emphasis(out)):
        return ""
    return out


# --------------------------------------------------------------------------
# Output side: normalizing messy LLM markdown
# --------------------------------------------------------------------------

@dataclass
class _Section:
    title: str
    body: List[str] = field(default_factory=list)  # markdown blocks


_TC = r"(?:\d{1,2}:)?\d{1,2}:\d{2}"
_RANGE_RE = re.compile(
    rf"[\[(]?\s*(?:с\s+)?(?P<a>{_TC})\s*(?:[-–—−]+|до|to)\s*(?P<b>{_TC})\s*[\])]?", re.IGNORECASE
)
_SINGLE_TC_RE = re.compile(rf"[\[(]\s*(?P<a>{_TC})\s*[\])]|(?P<b>{_TC})\s*$")
_LEADING_TC_RE = re.compile(rf"^(?:\[\s*{_TC}\s*\]|\(\s*{_TC}\s*\)|{_TC}(?=\s*[-–—:]?\s))\s*[-–—:]?\s*")
_INLINE_TC_RE = re.compile(rf"\s*[\[(]\s*(?:с\s+)?{_TC}(?:\s*(?:[-–—−]+|до)\s*{_TC})?\s*[\])]")
_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s*(.*?)(?:\s+#+)?\s*$")
_HR_RE = re.compile(r"^\s*([-*_=])(?:\s*\1){2,}\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-*+•▪●◦‣·–—])\s+(.*)$")
_NUMBERED_RE = re.compile(r"^\s*\d{1,3}[.)]\s+(.*)$")
_QUOTE_RE = re.compile(r"^\s*>+\s?(.*)$")
_BOLD_LINE_RE = re.compile(r"^\s*(?:\d{1,3}[.)]\s*)?\*\*([^*]+?)\*\*\s*:?\s*$")
_NUM_PREFIX_RE = re.compile(
    r"^(?:(?:раздел|часть|глава|тема|section|part)\s*(?:№\s*)?(?:\d+|(?-i:[IVXLC]+))(?:\.\d+)*\s*[.):—–-]?\s*"
    r"|(?:\d+(?:\.\d+)*|(?-i:[IVXLC]+))[.)]\s+)",
    re.IGNORECASE,
)
_LABEL_PREFIX_RE = re.compile(r"^(?:заголовок|название(?:\s+раздела)?|тема(?:\s+раздела)?)\s*[:—–-]\s*",
                              re.IGNORECASE)
_CALLOUT_LEAD_RE = re.compile(
    r"^\s*(?P<m>[*_]{0,3})\s*"
    r"(?:\[\s*уточнение\s*\]|уточнение\s+по\s+ходу\s+лекции|уточнение(?=\s*(?:\*\*|__|\*|_)?\s*[:—–-]))"
    r"\s*(?P=m)?\s*[:.—–-]?\s*(?P=m)?\s*",
    re.IGNORECASE,
)
_CLAR_MARK_RE = re.compile(r"\[\s*уточнение\s*\]", re.IGNORECASE)
_CHATTER_RE = re.compile(
    r"^(?:конечно|разумеется|хорошо|отлично|с удовольствием|sure|certainly)\s*[,!.:]"
    r"|^(?:вот|ниже)\b.{0,60}?(?:конспект|структур|раздел|разбивк|оформлен|представлен|краткое|продолжени)"
    r"|^надеюсь\b|^если\s+(?:нужно|хотите|потребуется|понадобится|у вас)"
    r"|^(?:дайте знать|обращайтесь|удачи|успехов)",
    re.IGNORECASE,
)
_TOC_TITLES = {
    "оглавление", "содержание", "план", "план лекции", "содержание лекции", "содержание фрагмента",
    "план фрагмента", "разделы", "смысловые разделы", "список разделов", "разделы фрагмента",
    "разделы лекции", "структура", "структура лекции", "структура фрагмента", "темы лекции",
    "table of contents", "contents", "outline",
}
# Model-made recaps of a chunk: a student doesn't write them, the lecture
# content is already in the notes above.
_RECAP_TITLES = {
    "вывод", "выводы", "итог", "итоги", "заключение", "резюме", "краткое содержание", "краткий итог",
    "общий вывод", "краткое резюме", "основные выводы", "ключевые моменты", "основные моменты",
    "основные идеи", "основные мысли", "ключевые идеи", "ключевые мысли", "основные тезисы",
    "ключевые тезисы", "главное", "кратко", "summary", "conclusion", "итоги лекции", "выводы лекции",
    "краткое содержание лекции", "заключение лекции",
}
_GENERIC_TITLE_RE = re.compile(
    r"^(?:(?:структурированный\s+|краткий\s+)?конспект(?:\s+\w+){0,4}"
    r"|фрагмент(?:\s+лекции)?|лекция|расшифровка(?:\s+фрагмента)?|продолжение(?:\s+\w+){0,2}|часть"
    r"|(?:продолжение|окончание)\s+(?:лекции|конспекта|темы))"
    r"(?:\s+(?:№\s*)?\d+(?:\s+из\s+\d+)?)?$",
    re.IGNORECASE,
)
_META_TITLE_RE = re.compile(
    r"^(?:(?:вступлени\w*|введение|приветствие|начало|окончание|завершение|конец|заключение)\s+"
    r"(?:лекци\w*|лектора|преподавателя|занятия|записи|фрагмента|разговора|конспекта)"
    r"|организационн\w+(?:\s+\w+)?|обращение\s+к\s+\w+|вопросы?\s+(?:из\s+(?:зала|аудитории)|студент\w*|"
    r"слушател\w*|аудитории)(?:\s+\w+){0,2}|ответы?\s+на\s+вопрос\w*(?:\s+\w+){0,2}|"
    r"неразборчив\w*(?:\s+\w+){0,2}|непонятн\w*(?:\s+\w+){0,2}|реплики(?:\s+\w+){0,2}|"
    r"комментари\w*(?:\s+\w+){0,2}|разное|прочее|приветствие|вступление|примечани\w*|"
    r"(?:\w+\s+)?(?:" + _LECT + r")(?:\s+\w+){0,2})$",
    re.IGNORECASE,
)
_CONTINUATION_RE = re.compile(r"\s*[(\[]?\s*(?:продолжение|окончание)\s*[)\]]?\s*$", re.IGNORECASE)
# Small models echo the format template and the instructions themselves;
# these never carry lecture content.
_PLACEHOLDER_RE = re.compile(
    r"^<?\s*(?:короткий\s+заголовок|заголовок\s+раздела|тема\s+раздела|название\s+темы)\b\s*>?\s*[:.—–-]?\s*",
    re.IGNORECASE,
)
_PLACEHOLDER_TITLE_RE = re.compile(
    r"^(?:ключевые\s+мысли|маркированный\s+список|короткий\s+заголовок|заголовок\s+раздела|тема\s+раздела"
    r"|название\s+темы|конспект\s+(?:этого\s+)?фрагмента|фрагмент\s+расшифровки|расшифровка)",
    re.IGNORECASE,
)
_TOPIC_LINE_RE = re.compile(
    r"^\**\s*(?:тема(?:\s+(?:лекции|занятия|фрагмента))?|название\s+лекции)\s*\**\s*[:—–-]\s*\**\s*(.+?)\s*\**\s*$",
    re.IGNORECASE,
)
_INSTRUCTION_ECHO_RE = re.compile(
    r"составь(?:те)?\s+конспект|начни(?:те)?\s+(?:ответ|со\s+строки)|по\s+правилам\s+выше|расшифровка\s+фрагмента\s*:"
    r"|^фрагмент\s+\d+\s+из\s+\d+|строго\s+по\s+образцу|уже\s+написанн|последние\s+строки"
    r"|расшифровка\s+(?:следующей\s+)?части|конспект\s+этой\s+части|продолжи(?:те)?\s+конспект"
    r"|пример\s+оформления|как\s+пишется\s+конспект|^запрещено\s*:"
    r"|<\s*(?:термин|определение\s+или\s+пояснение|ключевая\s+мысль|пересказ\s+ответа|тема\s+раздела|название)",
    re.IGNORECASE,
)
_ECHO_PREFIX_RE = re.compile(
    rf"^(?P<m>\**)\s*(?:#{{1,6}}\s*)?(?:(?:короткий\s+заголовок|заголовок\s+раздела)\b\s*[:.—–-]?\s*)?"
    rf"(?:\[\s*{_TC}(?:\s*[-–—]\s*{_TC})?\s*\]\s*)?",
    re.IGNORECASE,
)
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$")
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2700-\u27BF\u2B50\u2B55\uFE0F\u200d]")

# --- LaTeX -> plain text ----------------------------------------------------

_LATEX_SYMBOLS = {
    "cdot": "·", "cdotp": "·", "times": "×", "div": "÷", "pm": "±", "mp": "∓",
    "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "ne": "≠", "neq": "≠", "approx": "≈",
    "equiv": "≡", "sim": "~", "propto": "∝", "to": "→", "rightarrow": "→", "Rightarrow": "⇒",
    "implies": "⇒", "leftarrow": "←", "Leftarrow": "⇐", "leftrightarrow": "↔",
    "Leftrightarrow": "⇔", "iff": "⇔", "mapsto": "↦", "infty": "∞", "partial": "∂",
    "nabla": "∇", "int": "∫", "iint": "∬", "oint": "∮", "sum": "Σ", "prod": "Π",
    "forall": "∀", "exists": "∃", "in": "∈", "notin": "∉", "subset": "⊂", "subseteq": "⊆",
    "supset": "⊃", "cup": "∪", "cap": "∩", "emptyset": "∅", "varnothing": "∅", "neg": "¬",
    "land": "∧", "lor": "∨", "wedge": "∧", "vee": "∨", "circ": "∘", "degree": "°",
    "angle": "∠", "perp": "⊥", "parallel": "∥", "cdots": "…", "ldots": "…", "dots": "…",
    "prime": "′", "hbar": "ħ", "ell": "ℓ", "langle": "⟨", "rangle": "⟩", "vert": "|", "mid": "|",
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "θ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ",
    "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π",
    "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    "lim": "lim", "log": "log", "ln": "ln", "lg": "lg", "sin": "sin", "cos": "cos", "tan": "tan",
    "tg": "tg", "ctg": "ctg", "cot": "cot", "arcsin": "arcsin", "arccos": "arccos",
    "arctan": "arctan", "arctg": "arctg", "exp": "exp", "max": "max", "min": "min",
    "sup": "sup", "inf": "inf", "det": "det", "dim": "dim", "ker": "ker", "gcd": "gcd",
    "quad": " ", "qquad": " ", "displaystyle": "", "textstyle": "", "left": "", "right": "",
    "big": "", "Big": "", "bigg": "", "Bigg": "", "limits": "", "nolimits": "",
}
_LATEX_WRAPPERS = (
    "text|textrm|textbf|textit|mathrm|mathbf|mathit|mathsf|mathtt|mathcal|operatorname|"
    "boldsymbol|vec|hat|bar|overline|underline|tilde|widehat|dot|ddot"
)
_BLACKBOARD = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ"}


def _group(expr: str) -> str:
    expr = expr.strip()
    return expr if re.fullmatch(r"[\w.′]+", expr) else f"({expr})"


def _latex_to_plain(expr: str, in_math: bool = True) -> str:
    """Best-effort LaTeX → readable plain text ("\\frac{a}{b}" → "a/b").

    Outside math delimiters only known commands are touched, so a stray
    backslash (e.g. a Windows path) survives.
    """
    for _ in range(6):  # innermost constructs first; nested ones need several passes
        prev = expr
        expr = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
                      lambda m: f"{_group(m[1])}/{_group(m[2])}", expr)
        expr = re.sub(r"\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}",
                      lambda m: f"{_group(m[2])}^(1/{m[1].strip()})", expr)
        expr = re.sub(r"\\sqrt\s*\{([^{}]*)\}", lambda m: f"√{_group(m[1])}", expr)
        expr = re.sub(r"\\mathbb\s*\{\s*([A-Za-z])\s*\}", lambda m: _BLACKBOARD.get(m[1], m[1]), expr)
        expr = re.sub(rf"\\(?:{_LATEX_WRAPPERS})\s*\{{([^{{}}]*)\}}", r"\1", expr)
        if expr == prev:
            break
    expr = re.sub(r"\\lim_\s*\{([^{}]*)\}", r"lim \1 ", expr)
    if in_math:
        expr = re.sub(r"\^\s*\{([^{}]*)\}", lambda m: "^" + _group(m[1]), expr)
        expr = re.sub(r"_\s*\{([^{}]*)\}", lambda m: "_" + _group(m[1]), expr)
        expr = re.sub(r"\\[,;:!> ]", " ", expr)
        expr = expr.replace("\\\\", "; ")
        expr = re.sub(r"\\([{}%&#_$])", r"\1", expr)
        expr = re.sub(r"\\([A-Za-z]+)", lambda m: _LATEX_SYMBOLS.get(m[1], m[1]), expr)
        expr = expr.replace("{", "").replace("}", "")
    else:
        expr = re.sub(r"\\([A-Za-z]+)(?![A-Za-z])",
                      lambda m: _LATEX_SYMBOLS.get(m[1], m[0]), expr)
    return re.sub(r"[ \t]{2,}", " ", expr)


def _strip_latex(text: str) -> str:
    math_ = lambda m: _latex_to_plain(m.group(1)).strip()  # noqa: E731
    text = re.sub(r"\$\$(.+?)\$\$", math_, text, flags=re.S)
    text = re.sub(r"\\\[(.+?)\\\]", math_, text, flags=re.S)
    text = re.sub(r"\\\((.+?)\\\)", math_, text, flags=re.S)
    # Inline $…$: no space just inside the dollars and no word character just
    # outside them, so prices like "5$ и 10$" are left alone.
    text = re.sub(r"(?<![\w$\\])\$(?=\S)([^$\n]+?)(?<=\S)\$(?![\w$])", math_, text)
    if "\\" in text:
        text = _latex_to_plain(text, in_math=False)
    return text


# --- text-level cleanup -----------------------------------------------------

def _split_row(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _tables_to_lines(text: str) -> str:
    """Markdown tables → plain dash lines ("a — b"); the header row of a
    table with a separator line is column names, not content, and goes."""
    lines = text.split("\n")
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.count("|") >= 2 and i + 1 < len(lines) and _TABLE_SEP_RE.match(lines[i + 1]) \
                and "-" in lines[i + 1]:
            i += 2  # header + separator
            while i < len(lines) and lines[i].count("|") >= 1 and lines[i].strip():
                if not _TABLE_SEP_RE.match(lines[i]):
                    cells = [c for c in _split_row(lines[i]) if c]
                    if cells:
                        out.append("- " + " — ".join(cells))
                i += 1
            continue
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
            if _TABLE_SEP_RE.match(stripped):
                i += 1
                continue
            cells = [c for c in _split_row(stripped) if c]
            if cells:
                out.append("- " + " — ".join(cells))
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _normalize_text(raw: str) -> str:
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.S | re.IGNORECASE)
    if re.search(r"</think>", text, flags=re.IGNORECASE):
        # The template ate the opening tag: everything before the close is reasoning.
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    text = "\n".join(line for line in text.split("\n") if not re.match(r"^\s*(?:```|~~~)", line))
    text = _strip_latex(text)
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?\s*(?:b|strong)\s*>", "**", text, flags=re.IGNORECASE)
    text = re.sub(r"</?\s*(?:i|em)\s*>", "*", text, flags=re.IGNORECASE)
    text = re.sub(r"</?\s*(?:u|p|div|span|sup|sub|ul|ol|li|h[1-6]|mark|font|table|tr|td|th|thead|tbody)\b[^>]*>",
                  "", text, flags=re.IGNORECASE)
    text = text.replace("\t", "    ").replace("\u00a0", " ")
    text = _EMOJI_RE.sub("", text)
    return _tables_to_lines(text)


def _clean_inline(text: str) -> str:
    text = " ".join(text.split())
    text = re.sub(r"__(\S(?:.*?\S)?)__", r"**\1**", text)
    text = text.replace("`", "")
    if text.count("**") % 2:
        text = text.replace("**", "")
    text = re.sub(r"\*\*\s*\*\*", "", text)
    return text.strip()


def _norm_key(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower().replace("ё", "е")).split())


def _strip_emphasis(text: str) -> str:
    return re.sub(r"[*_`]+", "", text).strip()


def _is_toc_title(text: str) -> bool:
    return _norm_key(_strip_emphasis(text)) in _TOC_TITLES


def _is_recap_title(text: str) -> bool:
    return _norm_key(text) in _RECAP_TITLES


def _is_generic_title(text: str) -> bool:
    key = _norm_key(text)
    return not key or bool(_GENERIC_TITLE_RE.match(key) or _PLACEHOLDER_TITLE_RE.match(key))


def _is_meta_title(text: str) -> bool:
    key = _norm_key(text)
    return bool(key) and (bool(_META_TITLE_RE.match(key)) or not strip_meta_commentary(text))


def _similar(a: str, b: str) -> bool:
    """Two headings name the same topic ("Производная" / "Производная
    (продолжение)" / "Производная функции")."""
    ka = _norm_key(_CONTINUATION_RE.sub("", a or ""))
    kb = _norm_key(_CONTINUATION_RE.sub("", b or ""))
    if not ka or not kb:
        return False
    if ka == kb or (len(ka) >= 5 and len(kb) >= 5 and (ka in kb or kb in ka)):
        return True
    wa = {w[:5] for w in ka.split() if len(w) > 3}
    wb = {w[:5] for w in kb.split() if len(w) > 3}
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.6


def _parse_heading(text: str) -> str:
    """Heading text → clean title: numbering, emphasis, "Раздел 1:"
    prefixes, placeholders and any time range are taken out."""
    title = _strip_emphasis(text)
    m = _RANGE_RE.search(title)
    if m:
        title = title[:m.start()] + " " + title[m.end():]
    else:
        m = _SINGLE_TC_RE.search(title)
        if m:
            title = title[:m.start()] + " " + title[m.end():]
    # Leftover brackets: empty ones or mangled timecodes like "[02:54:01:46]".
    title = re.sub(r"[\[(]\s*(?:\d[\d:\s.,–—-]*)?[\])]", " ", title)
    title = " ".join(title.split())
    for _ in range(2):
        title = _NUM_PREFIX_RE.sub("", title).strip()
        title = _LABEL_PREFIX_RE.sub("", title).strip()
        title = _PLACEHOLDER_RE.sub("", title).strip()
    title = title.strip(" :;,-–—<>").strip("«»\"'“”").strip()
    if len(title) > _MAX_TITLE_CHARS:
        # A whole transcript sentence instead of a title: keep its first clause.
        sentence = re.split(r"(?<=[.!?])\s", title, maxsplit=1)[0]
        if len(sentence) <= _MAX_TITLE_CHARS:
            title = sentence.rstrip(".")
        else:
            cut = title[:_MAX_TITLE_CHARS].rsplit(" ", 1)[0]
            title = cut.rstrip(" ,;:-–—") + "…"
    return title.rstrip(".")


def _callout_text(text: str) -> Optional[str]:
    """Body of a clarification-style line ("[УТОЧНЕНИЕ] …", "Уточнение по
    ходу лекции: …", "**Уточнение:** …") — old notes format, still produced
    by models that saw it somewhere — else None."""
    stripped = _LEADING_TC_RE.sub("", text.strip())
    m = _CALLOUT_LEAD_RE.match(stripped)
    if m:
        rest = stripped[m.end():]
    elif _CLAR_MARK_RE.search(stripped):
        rest = _CLAR_MARK_RE.sub(" ", stripped)
    else:
        return None
    rest = _LEADING_TC_RE.sub("", rest.strip())
    rest = re.sub(r"^[:.—–-]\s*", "", rest)
    return _clean_inline(rest)


def _classify(line: str) -> Tuple[str, str]:
    """One body line → (kind, text) with kind in blank/skip/h3/ul/ol/p."""
    s = line.strip()
    if not s:
        return "blank", ""
    if _HR_RE.match(s):
        return "skip", ""
    m = _MD_HEADING_RE.match(s)
    if m and m.group(2).strip():
        raw = m.group(2)
        if _INSTRUCTION_ECHO_RE.search(_strip_emphasis(raw)):
            return "skip", ""
        callout = _callout_text(raw)
        if callout is not None:
            s = _capitalize(callout)  # a clarification written as a heading: an ordinary line
        else:
            title = _parse_heading(raw)
            if (not title or _is_toc_title(title) or _is_generic_title(title) or _is_recap_title(title)
                    or _is_meta_title(title)):
                return "skip", ""
            return "h3", title
    m = _QUOTE_RE.match(s)
    quoted = bool(m)
    if m:
        s = m.group(1).strip()  # students don't write block quotes: the words become a line
        if not s:
            return "blank", ""
    kind = "ul" if quoted else "p"
    m = _BULLET_RE.match(s)
    if m:
        kind, s = "ul", m.group(1)
    else:
        m = _NUMBERED_RE.match(s)
        if m:
            kind, s = "ol", m.group(1)
    if _INSTRUCTION_ECHO_RE.search(_strip_emphasis(s)):
        return "skip", ""
    if kind == "p" and _BOLD_LINE_RE.match(s) and len(_strip_emphasis(s)) <= _MAX_TITLE_CHARS:
        # A standalone "**Свойства пределов:**" line introduces what follows.
        title = _parse_heading(s)
        if _callout_text(s) is None and title and not _is_generic_title(title):
            if _is_toc_title(title) or _is_recap_title(title) or _is_meta_title(title):
                return "skip", ""
            return "h3", title
    # "**## Короткий заголовок [01:47] text**" → "text"
    m = _ECHO_PREFIX_RE.match(s)
    echoed = bool(m and m.end() > len(m.group("m")))
    if echoed:
        s = m.group("m") + s[m.end():]
    callout = _callout_text(s)
    if callout is not None:
        s = _capitalize(callout)
        if kind == "p":
            kind = "ul"
    if _LEADING_TC_RE.match(s) or (echoed and kind == "p"):
        # The model echoed transcript lines; keep the words, drop the timecode.
        s = _capitalize(_LEADING_TC_RE.sub("", s))
        if kind == "p":
            kind = "ul"
    s = _INLINE_TC_RE.sub("", s)
    s = _clean_inline(s)
    if (echoed or len(s) > 100) and s.startswith("**") and s.endswith("**") and s.count("**") == 2:
        s = s[2:-2].strip()  # a fully bold (echoed or long) item is a formatting slip, not emphasis
    s = strip_meta_commentary(s)
    s = re.sub(r"(?<!\d)\$", "", s).strip()  # prices keep it, LaTeX leftovers don't
    return (kind, s) if s else ("skip", "")


def _blocks_from_lines(lines: Sequence[str], seen: Optional[set] = None) -> List[str]:
    """Body lines → markdown blocks of the subset (lists grouped, every
    paragraph / subheading its own block).

    ``seen`` (shared across the whole lecture) drops verbatim repeats:
    small models loop, and they echo the tail of the notes they were given.
    """
    blocks: List[str] = []
    list_kind: Optional[str] = None
    list_items: List[str] = []

    def close_list() -> None:
        nonlocal list_kind, list_items
        if list_items:
            if list_kind == "ol":
                blocks.append("\n".join(f"{i}. {t}" for i, t in enumerate(list_items, 1)))
            else:
                blocks.append("\n".join(f"- {t}" for t in list_items))
        list_kind, list_items = None, []

    if seen is None:
        seen = set()
    for line in lines:
        kind, text = _classify(line)
        if kind in ("skip", "blank"):
            continue  # lists stay open across blank lines ("loose" lists)
        key = (kind == "h3", _norm_key(text))
        if key in seen:
            continue
        seen.add(key)
        if kind in ("ul", "ol"):
            if list_kind != kind:
                close_list()
                list_kind = kind
            list_items.append(text)
            continue
        close_list()
        blocks.append(f"### {text}" if kind == "h3" else text)
    close_list()
    # Subheadings with nothing under them (left after dedupe/echo removal).
    return [b for i, b in enumerate(blocks)
            if not (b.startswith("### ") and (i + 1 == len(blocks) or blocks[i + 1].startswith("### ")))]


def _is_chatter_block(block: str) -> bool:
    if "\n" in block:
        return False
    return bool(_CHATTER_RE.match(_strip_emphasis(re.sub(r"^(?:[-*]|\d+\.)\s+", "", block))))


def _is_label_line(line: str) -> bool:
    """Short "Конспект фрагмента:"-style lines that only introduce what follows."""
    s = _strip_emphasis(line)
    return 0 < len(s) <= 80 and s.endswith(":") and not _BULLET_RE.match(line) and not _NUMBERED_RE.match(line)


def _drop_preamble_toc(lines: List[str], section_titles: Sequence[str]) -> List[str]:
    """Remove an unheaded table of contents ("Оглавление:" + list, or a list
    that merely repeats the section titles) and chatter from the preamble."""
    keys = {_norm_key(t) for t in section_titles if t}
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = _strip_emphasis(line).rstrip(":").strip()
        if _is_toc_title(stripped):
            i += 1
            while i < len(lines) and (not lines[i].strip() or _BULLET_RE.match(lines[i])
                                      or _NUMBERED_RE.match(lines[i]) or _RANGE_RE.search(lines[i])):
                i += 1
            continue
        if _BULLET_RE.match(line) or _NUMBERED_RE.match(line):
            j = i
            items = []
            while j < len(lines) and (_BULLET_RE.match(lines[j]) or _NUMBERED_RE.match(lines[j])):
                m = _BULLET_RE.match(lines[j]) or _NUMBERED_RE.match(lines[j])
                items.append(m.group(1))
                j += 1
            hits = ranged = 0
            for item in items:
                key = _norm_key(_parse_heading(item))
                if key and any(key == k or key in k or k in key for k in keys):
                    hits += 1
                if _RANGE_RE.search(item):
                    ranged += 1
            if keys and (hits * 2 >= len(items) or ranged == len(items)):
                i = j
                continue
            out.extend(lines[i:j])
            i = j
            continue
        if (line.strip() and not any(o.strip() for o in out)
                and (_is_chatter_block(line.strip()) or _is_label_line(line))):
            i += 1  # leading "Вот конспект фрагмента:"-style lines
            continue
        out.append(line)
        i += 1
    return out


def _split_sections(text: str) -> Tuple[List[str], list]:
    """Normalized answer → (preamble lines, [(title, body lines, raw heading)])."""
    lines = text.split("\n")
    heads = []  # (line index, level, raw heading text)
    for idx, line in enumerate(lines):
        m = _MD_HEADING_RE.match(line)
        if m and m.group(2).strip() and not line.lstrip().startswith("#!"):
            heads.append((idx, len(m.group(1)), m.group(2).strip()))
    if not heads:
        # No markdown headings: small models often write topic titles as
        # "**1. Title**" or "1. Title [00:00–02:10]" lines instead. (Plain bold
        # lines become subheadings in _classify.)
        for idx, line in enumerate(lines):
            if _QUOTE_RE.match(line) or _BULLET_RE.match(line) or _callout_text(line) is not None:
                continue
            stripped = line.strip()
            numbered_bold = bool(_BOLD_LINE_RE.match(line)) and bool(
                re.match(r"^(?:\d{1,3}[.)]\s*\**|\**\s*\d{1,3}[.)]\s)", stripped))
            if numbered_bold and len(stripped) <= _MAX_TITLE_CHARS + 10:
                heads.append((idx, 2, stripped))
            elif (_RANGE_RE.search(stripped) and len(stripped) <= 150
                  and not _LEADING_TC_RE.match(stripped)):
                heads.append((idx, 2, stripped))

    real = [h for h in heads if not _is_toc_title(_parse_heading(h[2]) or h[2])]
    dropped = set()
    level1 = [h for h in real if h[1] == 1]
    deeper = [h for h in real if h[1] > 1]
    if level1 and deeper:
        # A lone "# Конспект лекции" on top is a document title, not a topic.
        lone_title = len(level1) == 1 and real[0][1] == 1
        if lone_title or all(_is_generic_title(_parse_heading(h[2])) for h in level1):
            dropped = {h[0] for h in level1}
    levels = [h[1] for h in real if h[0] not in dropped]
    section_level = min(levels) if levels else None

    head_at = {h[0]: h for h in heads}
    preamble: List[str] = []
    sections: List[Tuple[str, List[str], str]] = []
    skipping_toc = False
    for idx, line in enumerate(lines):
        head = head_at.get(idx)
        if head is not None:
            _i, level, raw = head
            if idx in dropped:
                continue
            title = _parse_heading(raw)
            if _is_toc_title(title or raw):
                skipping_toc = True
                continue
            if section_level is not None and level <= section_level:
                skipping_toc = False
                sections.append((title, [], raw))
                continue
            if skipping_toc:
                continue
            line = f"### {raw}"  # deeper heading → subheading
        elif skipping_toc:
            if (not line.strip() or _BULLET_RE.match(line) or _NUMBERED_RE.match(line)
                    or _RANGE_RE.search(line)):
                continue
            skipping_toc = False
        (sections[-1][1] if sections else preamble).append(line)
    return preamble, sections


def _extend_blocks(body: List[str], blocks: Sequence[str]) -> None:
    """Append blocks, joining adjacent lists of the same kind into one list."""
    for block in blocks:
        if body and block.startswith("- ") and body[-1].startswith("- "):
            body[-1] = f"{body[-1]}\n{block}"
        elif body and re.match(r"^\d+\. ", block) and re.match(r"^\d+\. ", body[-1]):
            start = len(body[-1].split("\n"))
            items = [re.sub(r"^\d+\. ", "", it) for it in block.split("\n")]
            body[-1] += "".join(f"\n{start + i}. {t}" for i, t in enumerate(items, 1))
        else:
            body.append(block)


def _is_content_heading(raw: str) -> bool:
    """A "heading" that is really a transcript sentence (weak models put
    whole lines after "##")."""
    text = _strip_emphasis(raw)
    return len(text) > 80 or bool(re.search(r"[.!?]\s+\S", text))


def _drop_truncated_tail(text: str) -> str:
    """Drop the last line when it is a cut-off copy of an earlier one.

    A looping model runs into ``num_predict`` mid-repeat and leaves a
    fragment like "Вопросы, которые могут возникнуть в процессе изуче";
    the full line is already there, so the fragment is pure noise.
    """
    lines = text.rstrip().split("\n")
    if len(lines) < 2:
        return text
    last = _norm_key(lines[-1])
    if len(last) < 8:
        return text
    for line in lines[:-1]:
        key = _norm_key(line)
        if key != last and key.startswith(last):
            return "\n".join(lines[:-1])
    return text


def _clean_topic(text: str) -> str:
    topic = _parse_heading(text)
    m = re.match(r"^(?:конспект(?:\s+лекции)?|лекция)\s*(?:(?:по\s+теме|на\s+тему)\s*[:—–-]?|[:—–-])\s*(.+)$",
                 topic, flags=re.IGNORECASE)
    if m:
        topic = m.group(1)
    elif re.match(r"^(?:конспект|лекция)\b", topic, flags=re.IGNORECASE):
        return ""  # "Конспект лекции по истории" names the genre, not the topic
    topic = topic.strip(" «»\"")
    topic = strip_meta_commentary(_clean_inline(topic)).replace("**", "").rstrip(".").strip()
    if not topic or len(topic) > _MAX_TOPIC_CHARS or _is_generic_title(topic) or _is_meta_title(topic):
        return ""
    return topic[:1].upper() + topic[1:]


def _take_topic(text: str) -> Tuple[str, str]:
    """(topic, rest): a leading "Тема: …" line (the first prompt asks for
    it), or a lone "# Title" on top of an answer with "##" headings."""
    lines = text.split("\n")
    idx = None
    for i, line in enumerate(lines[:40]):
        s = line.strip()
        if not s or _HR_RE.match(s):
            continue
        hm = _MD_HEADING_RE.match(s)
        inner = hm.group(2).strip() if hm else s
        if _TOPIC_LINE_RE.match(_strip_emphasis(inner)) or (
                re.match(r"^(?:конспект|лекция)\b", _strip_emphasis(inner), re.IGNORECASE) and _clean_topic(inner)):
            idx = i
            break
        # Skip what may precede it: chatter, "Конспект:" labels, "# Конспект лекции".
        if _is_chatter_block(s) or _is_label_line(s) or (hm and _is_generic_title(_parse_heading(inner))):
            continue
        idx = i
        break
    if idx is None:
        return "", text
    first = lines[idx].strip()
    m = _MD_HEADING_RE.match(first)
    inner = m.group(2).strip() if m else first
    tm = _TOPIC_LINE_RE.match(_strip_emphasis(inner)) or _TOPIC_LINE_RE.match(inner)
    if tm:
        return _clean_topic(tm.group(1)), "\n".join(lines[:idx] + lines[idx + 1:])
    rest = lines[idx + 1:]
    if (m and len(m.group(1)) == 1 and any(re.match(r"^\s{0,3}#{2,3}\s*\S", line) for line in rest)
            and not any(re.match(r"^\s{0,3}#(?!#)\s*\S", line) for line in rest)):
        return _clean_topic(inner), "\n".join(lines[:idx] + lines[idx + 1:])
    if re.match(r"^\**\s*конспект\b", _strip_emphasis(inner), re.IGNORECASE) and len(inner) <= _MAX_TOPIC_CHARS:
        return _clean_topic(inner), "\n".join(lines[:idx] + lines[idx + 1:])
    return "", text


@dataclass
class _ChunkNotes:
    topic: str
    lead: List[str]               # blocks before the first heading: continue the previous topic
    sections: List[_Section]


def _parse_chunk_answer(raw: str, seen: Optional[set] = None) -> _ChunkNotes:
    """One LLM answer → topic line (if any), leading blocks and titled
    sections, all inside the markdown subset and free of meta-commentary."""
    if seen is None:
        seen = set()
    text = _drop_truncated_tail(_normalize_text(raw))
    topic, text = _take_topic(text)
    preamble, parsed = _split_sections(text)
    preamble = _drop_preamble_toc(preamble, [p[0] for p in parsed])
    lead = _blocks_from_lines(preamble, seen)

    real_sections = [p for p in parsed if not _is_recap_title(p[0])]
    sections: List[_Section] = []
    for title, body, raw_heading in parsed:
        blocks = _blocks_from_lines(body, seen)
        if not blocks and _is_content_heading(raw_heading):
            # Keep the words of a body-less "content heading" as a line of
            # the topic it belongs to.
            blocks, title = _blocks_from_lines([f"- {raw_heading}"], seen), ""
        if _is_recap_title(title):
            if real_sections or lead:
                continue  # a recap of what is already written above
            title = ""
        if not blocks:
            continue  # heading-only (e.g. a TOC written as headings)
        if (_is_generic_title(title) or _is_meta_title(title) or _is_toc_title(title)
                or _INSTRUCTION_ECHO_RE.search(title)):
            title = ""
        if not title and blocks:
            m = _TOPIC_LINE_RE.match(blocks[0])
            if m and "\n" not in blocks[0]:
                title = _clean_topic(m.group(1))
                if title:
                    del blocks[0]
        if not blocks:
            continue
        if not title:
            if sections:
                _extend_blocks(sections[-1].body, blocks)
            else:
                _extend_blocks(lead, blocks)
            continue
        sections.append(_Section(title, blocks))

    while lead and _is_chatter_block(lead[0]):
        lead.pop(0)
    tail_body = sections[-1].body if sections else lead
    while tail_body and _is_chatter_block(tail_body[-1]):
        tail_body.pop()
    sections = [sec for sec in sections if sec.body]
    return _ChunkNotes(topic, lead, sections)


def _merge_continuations(sections: List[_Section]) -> List[_Section]:
    """Join neighbouring sections about the same topic ("Производная" +
    "Производная (продолжение)") into one."""
    out: List[_Section] = []
    for sec in sections:
        if out and sec.title and out[-1].title and _similar(sec.title, out[-1].title):
            _extend_blocks(out[-1].body, sec.body)
            continue
        out.append(sec)
    return out


_FINAL_TC_RE = re.compile(rf"\s*[\[(]\s*{_TC}(?:\s*[-–—−]+\s*{_TC})?\s*[\])]")


def _assemble(topic: str, sections: Sequence[_Section]) -> str:
    parts = []
    if topic:
        parts.append(f"**Тема:** {topic}")
    for sec in sections:
        body = "\n\n".join(sec.body).strip()
        if not body:
            continue
        parts.append(f"## {sec.title}\n\n{body}" if sec.title else body)
    md = "\n\n".join(parts)
    md = _FINAL_TC_RE.sub("", md)  # belt and braces: no timecodes in notes, ever
    return re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

def _notes_tail(sections: Sequence[_Section], limit: int = TAIL_CHARS) -> str:
    """Last heading + the last lines under it (≤ ``limit`` chars): what the
    next call continues from (photo lines are ours, not the model's: left out)."""
    if not sections:
        return ""
    last = sections[-1]
    lines = [line for line in "\n\n".join(last.body).split("\n")
             if line.strip() and not _IMAGE_LINE_RE.match(line)]
    take: List[str] = []
    total = 0
    for line in reversed(lines):
        if take and total + len(line) > limit:
            break
        if len(line) > limit:
            line = "…" + line[-limit:].split(" ", 1)[-1]
        take.append(line)
        total += len(line) + 1
    take.reverse()
    head = [f"## {last.title}"] if last.title else []
    return "\n".join(head + take)


def _photo_context(texts: Sequence[str], limit: int = PHOTO_PROMPT_MAX_CHARS) -> str:
    """Photo text for one chunk's prompt, within ``limit`` chars in total."""
    kept: List[str] = []
    budget = limit
    for text in texts:
        text = (text or "").strip()
        if not text or budget < 80:
            continue
        if len(text) > budget:
            text = text[:budget - 1].rsplit("\n", 1)[0].rstrip() + "…"
        kept.append(text)
        budget -= len(text)
    return "\n---\n".join(kept)


def _chunk_prompt(chunk_text: str, index: int, tail: str, topic: str, title_hint: str,
                  photo_text: str = "") -> str:
    parts = [
        "Ты — студент на лекции и ведёшь конспект. Ниже — расшифровка речи лектора, сделанная "
        "автоматическим распознаванием речи (возможны ошибки).",
        "",
        _RULES,
        "",
        _EXAMPLE,
        "",
    ]
    if title_hint:
        parts.append(f"Название записи (может подсказывать тему): {title_hint}")
    if index == 0 or not tail:
        # Without the explicit "## …" here T-lite sometimes writes the whole
        # lecture as one flat list of 25+ items (seen on a real recording).
        parts.append("Начни ответ со строки «Тема: …» — тема лекции в 2–6 словах, затем заголовок первой "
                     "темы строкой «## …» и пункты под ним.")
    else:
        if topic:
            parts.append(f"Тема лекции: {topic}")
        parts += [
            "Конспект уже начат. Вот его последние строки (не повторяй их):",
            "---",
            tail,
            "---",
            "Продолжи конспект с этого места. Если лектор продолжает ту же тему — сразу пиши пункты, "
            "без заголовка; новый заголовок «## » — только для новой темы.",
        ]
    if photo_text:
        # Board/slide text is the most reliable source for formulas, terms
        # and names the recognizer garbles; it's context, not a second text
        # to summarize.
        parts += [
            "",
            "Во время этой части лекции показывали слайд или доску. Текст с фото (распознан автоматически) — "
            "сверяй по нему формулы, термины и имена, но конспектируй речь лектора:",
            "---",
            photo_text,
            "---",
        ]
    parts += [
        "",
        "Расшифровка следующей части лекции:",
        "---",
        chunk_text,
        "---",
        "",
        "Напиши конспект этой части. Только конспект, без пояснений.",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Photos
# --------------------------------------------------------------------------

def photo_markdown(photo: "NotesPhoto") -> str:
    """The notes line of one photo: ``![Фото 1](photos/001.jpg)`` (export.py
    turns it into the image with its caption under it)."""
    caption = " ".join((photo.caption or "Фото").replace("[", "(").replace("]", ")").split())
    return f"![{caption}]({photo.src})"


def photos_section(photos: Sequence["NotesPhoto"]) -> str:
    """«Фото с занятия» section with the given photos ("" for none)."""
    lines = [photo_markdown(p) for p in photos]
    if not lines:
        return ""
    return f"## {PHOTOS_SECTION_TITLE}\n\n" + "\n\n".join(lines)


def append_photos(markdown: str, photos: Sequence["NotesPhoto"]) -> str:
    """Notes with a «Фото с занятия» section at the end (unchanged without photos)."""
    section = photos_section(photos)
    if not section:
        return markdown
    return (markdown or "").rstrip() + "\n\n" + section + "\n"


def _assign_photos(chunks: Sequence[Sequence[Tagged]], photos: Sequence["NotesPhoto"]):
    """{chunk index: photos taken while it was spoken} and the photos without
    a time. A photo belongs to the last chunk that started at or before it
    (a photo taken in a pause goes with what was said just before). This
    is the chunk whose prompt gets the photo text; see _place_photos for
    where the image goes."""
    starts = [float(chunk[0][0].start) for chunk in chunks]
    by_chunk = {}
    untimed = []
    for photo in photos:
        if photo.offset_s is None or not starts:
            untimed.append(photo)
            continue
        index = 0
        for i, start in enumerate(starts):
            if start <= photo.offset_s:
                index = i
        by_chunk.setdefault(index, []).append(photo)
    return by_chunk, untimed


def _place_photos(chunks: Sequence[Sequence[Tagged]], photos: Sequence["NotesPhoto"]) -> dict:
    """{chunk index: timed photos whose image goes right after that chunk's
    notes}. The chunk boundary nearest to the photo's moment wins (the end
    of the last chunk is its last segment's end), so the image sits where
    it was taken in the talk rather than up to a chunk's length later."""
    bounds = [float(chunk[0][0].start) for chunk in chunks[1:]]
    if chunks:
        last = chunks[-1][-1][0]
        bounds.append(float(getattr(last, "end", last.start)))
    placed: dict = {}
    for photo in photos:
        if photo.offset_s is None or not bounds:
            continue
        index = min(range(len(bounds)), key=lambda i: (abs(bounds[i] - photo.offset_s), i))
        placed.setdefault(index, []).append(photo)
    return placed


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _check_cancel(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled()


class _Progress:
    """Monotonic, throttled fraction reporting (token streaming calls it a lot)."""

    def __init__(self, callback: Optional[Callable[[float], None]], units: float) -> None:
        self._callback = callback
        self._units = max(units, 1e-9)
        self._last = 0.0

    def report(self, done_units: float, force: bool = False) -> None:
        if self._callback is None:
            return
        frac = max(0.0, min(1.0, done_units / self._units))
        if frac < self._last or (not force and frac - self._last < 0.01 and frac < 1.0):
            return
        self._last = frac
        self._callback(frac)


def _generate(llm, model: str, prompt: str, system: str, options: dict,
              cancel_event: Optional[threading.Event], on_token=None, what: str = "",
              keep_alive: Optional[str] = None) -> str:
    """One LLM call with a single retry on errors or an empty answer."""
    extra = {} if keep_alive is None else {"keep_alive": keep_alive}
    for attempt in (1, 2):
        _check_cancel(cancel_event)
        try:
            result = llm.stream_generate(model=model, prompt=prompt, system=system, on_token=on_token,
                                         options=options, cancel_event=cancel_event, **extra)
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001 — retried once, then re-raised as is
            _check_cancel(cancel_event)
            if attempt == 2:
                raise
            logger.warning("LLM call for %s failed (%s: %s), retrying once", what, type(exc).__name__, exc)
            if cancel_event is not None:
                cancel_event.wait(1.0)
            continue
        _check_cancel(cancel_event)
        text = result or ""
        if text.strip() or attempt == 2:
            return text
        logger.warning("LLM returned an empty answer for %s, retrying once", what)
    return ""


def structure_lecture(
    tagged: Sequence[Tagged],
    llm,
    llm_model: str,
    on_progress: Optional[Callable[[float], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    options: Optional[dict] = None,
    title_hint: str = "",
    chunk_chars: Optional[int] = None,
    photos: Sequence["NotesPhoto"] = (),
    photo_placement: str = "timestamps",
    before_chunk: Optional[Callable[[int], Optional[Tuple[str, dict]]]] = None,
    keep_alive: Optional[str] = None,
    last_keep_alive: Optional[str] = None,
) -> str:
    """Student-style lecture notes (markdown subset) from tagged lecturer segments.

    ``llm`` is an OllamaClient (anything with ``stream_generate``). Raises
    Cancelled when ``cancel_event`` is set; LLM errors that persist after a
    retry propagate unchanged. ``on_progress(fraction)`` is called during
    and after every chunk. ``chunk_chars`` overrides the transcript size of
    one call (bigger contexts can take more).

    ``photos``: the recording's photos. A timed photo's text is context for
    the chunk it was taken in; ``photo_placement`` "timestamps" first cuts
    the chunks at the photo moments (split_at_moments) and puts each image
    after the notes of the chunk boundary nearest to its moment (untimed
    ones at the end), "end" collects all of them in a «Фото с занятия»
    section at the end.

    ``before_chunk(i)`` runs before every chunk (the RAM guard re-checks
    there) and may return a replacement (model, options) for this and the
    following chunks. ``keep_alive`` / ``last_keep_alive``: how long Ollama
    keeps the model after each / the last chunk (None = the client default);
    "0" on the last one gives its RAM back right away.
    """
    options = dict(options or DEFAULT_LLM_OPTIONS)
    photos = list(photos or ())
    items = [(seg, clar) for seg, clar in tagged if (seg.text or "").strip()]
    if not items:
        if on_progress:
            on_progress(1.0)
        return append_photos(EMPTY_NOTES_MARKDOWN, photos)
    if not has_lecture_material(items):
        if on_progress:
            on_progress(1.0)
        return append_photos(NO_CONTENT_NOTES_MARKDOWN, photos)

    chunks = chunk_tagged(items, max_chars=chunk_chars or DEFAULT_CHUNK_CHARS)
    inline_photos = photo_placement != "end"
    if inline_photos:
        chunks = split_at_moments(chunks, [p.offset_s for p in photos if p.offset_s is not None])
    photos_by_chunk, untimed_photos = _assign_photos(chunks, photos)
    placed_photos = _place_photos(chunks, photos) if inline_photos else {}
    n = len(chunks)
    progress = _Progress(on_progress, n)
    seen: set = {(line.startswith("#"), _norm_key(_classify(line)[1])) for line in _EXAMPLE_LINES}
    topic = ""
    sections: List[_Section] = []
    for i, chunk in enumerate(chunks):
        _check_cancel(cancel_event)
        chunk_text = "\n".join(_format_line(seg, clar) for seg, clar in chunk)
        tail = _notes_tail(sections)
        chunk_photos = photos_by_chunk.get(i, [])
        prompt = _chunk_prompt(chunk_text, i, tail, topic or title_hint, title_hint,
                               photo_text=_photo_context([p.text for p in chunk_photos]))
        expected = max(400.0, 0.35 * len(chunk_text))
        produced = [0]

        def on_token(token: str, _i=i, _expected=expected, _produced=produced) -> None:
            _produced[0] += len(token or "")
            progress.report(_i + min(0.9, _produced[0] / _expected))

        if before_chunk is not None:
            replacement = before_chunk(i)
            if replacement is not None:
                llm_model, options = replacement[0], dict(replacement[1] or options)
        t0 = time.monotonic()
        raw = _generate(llm, llm_model, prompt, STRUCTURING_SYSTEM_PROMPT, options, cancel_event,
                        on_token=on_token, what=f"chunk {i + 1}/{n}",
                        keep_alive=last_keep_alive if i == n - 1 and last_keep_alive is not None else keep_alive)
        part = _parse_chunk_answer(raw, seen)
        if part.topic:
            if not topic and not sections:
                topic = part.topic
            elif not _similar(part.topic, topic):
                # A later "Тема: …" line names the next topic, not the lecture's.
                part.sections.insert(0, _Section(part.topic, part.lead))
                part.lead = []
        if part.lead:
            if sections:
                _extend_blocks(sections[-1].body, part.lead)
            else:
                sections.append(_Section("", list(part.lead)))
        for sec in part.sections:
            if sections and _similar(sec.title, sections[-1].title):
                _extend_blocks(sections[-1].body, sec.body)  # same topic across the boundary
            else:
                sections.append(sec)
        if not part.lead and not part.sections:
            logger.warning("Chunk %d/%d: no usable notes in the answer (%d chars)", i + 1, n, len(raw))
        here = sorted(placed_photos.get(i, []), key=lambda p: p.offset_s)
        if here:
            image_blocks = [photo_markdown(p) for p in here]
            if sections:
                sections[-1].body.extend(image_blocks)
            else:
                sections.append(_Section("", image_blocks))
        logger.info("Structured chunk %d/%d: %d chars in, %d chars out, %d section(s), %.1f s",
                    i + 1, n, len(chunk_text), len(raw), len(part.sections), time.monotonic() - t0)
        logger.debug("Chunk %d raw answer:\n%s", i + 1, raw[:4000])
        progress.report(i + 1, force=True)

    sections = [sec for sec in _merge_continuations(sections) if sec.body]
    end_photos = untimed_photos if inline_photos else photos
    if not any(not _IMAGE_LINE_RE.match(block) for sec in sections for block in sec.body):
        _check_cancel(cancel_event)
        return append_photos(NO_CONTENT_NOTES_MARKDOWN, photos)
    if not topic and re.search(r"[А-Яа-яЁё]{4,}", title_hint or ""):
        topic = _clean_topic(title_hint)  # an imported file's name often names the topic
    return append_photos(_assemble(topic, sections), end_photos)
