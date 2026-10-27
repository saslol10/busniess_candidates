"""Локальные аналитические заключения через Ollama API.

Модуль не рассчитывает признаки и не устанавливает факты. Он передаёт
в локальную модель только уже рассчитанные приложением данные.
"""
import json
import os
import urllib.error
import urllib.request
import re

OLLAMA_ROOT = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_URL = OLLAMA_ROOT + "/api/generate"
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")


def is_ollama_online():
    try:
        request = urllib.request.Request(OLLAMA_ROOT + "/", method="GET")
        with urllib.request.urlopen(request, timeout=1.5) as response:
            return response.status < 500
    except Exception:
        return False


def _value(row, key, default="нет данных"):
    value = row.get(key, default) if hasattr(row, "get") else default
    try:
        # pandas NaN without importing numpy/pandas here.
        if value != value:
            return default
    except Exception:
        pass
    return value


def _state_text(states):
    if not states:
        return "- Сведения о статусах направлений не переданы."
    lines = []
    for state in states:
        name = state.get("Направление", state.get("Код", "направление"))
        status = state.get("Статус", "неизвестно")
        message = state.get("Сообщение", "")
        lines.append(f"- {name}: {status}" + (f" — {message}" if message else ""))
    return "\n".join(lines)


def _evidence_text(evidence_by_group):
    """Сжимает основания, чтобы не отправлять в модель весь Excel."""
    if not evidence_by_group:
        return "- Подробные строки оснований не переданы."
    lines = []
    for group, frame in evidence_by_group.items():
        if frame is None:
            continue
        try:
            if getattr(frame, "empty", True):
                continue
            records = frame.head(12).to_dict("records")
        except Exception:
            continue
        lines.append(f"{group}: найдено строк для кандидата — не менее {len(records)} в переданном фрагменте")
        for record in records:
            clean = []
            for key, value in record.items():
                if key in {"ИНН кандидата", "candidate_inn"}:
                    continue
                if value is None:
                    continue
                try:
                    if value != value:
                        continue
                except Exception:
                    pass
                clean.append(f"{key}={value}")
            if clean:
                lines.append("  " + "; ".join(clean[:10]))
    return "\n".join(lines) if lines else "- Строки оснований для кандидата отсутствуют."



def _role_evidence_text(evidence_by_group):
    """Возвращает подробные основания по учредителям и руководителям для Qwen."""
    if not evidence_by_group:
        return "- Подробные основания по учредителям и руководителям не переданы."

    sections = []
    role_groups = (("founders", "Учредители"), ("directors", "Руководители"))
    for group, title in role_groups:
        frame = evidence_by_group.get(group)
        if frame is None:
            continue
        try:
            if getattr(frame, "empty", True):
                continue
            records = frame.to_dict("records")
        except Exception:
            continue

        lines = [f"{title}:"]
        for record in records:
            lines.append(
                "- ФИО: {fio}; ИНН физлица: {inn}; "
                "роль в исходной организации: {src}; "
                "роль в организации-кандидате: {cand}; "
                "период: {period}; основание: {reason}.".format(
                    fio=_value(record, "ФИО"),
                    inn=_value(record, "ИНН физлица"),
                    src=_value(record, "Роль в исходной организации"),
                    cand=_value(record, "Роль в организации-кандидате"),
                    period=_value(record, "Период (для сотрудников)"),
                    reason=_value(record, "Основание"),
                )
            )
        # Исправлено: раньше было "\\n" — в промпт попадал литерал \n.
        sections.append("\n".join(lines))

    if not sections:
        return "- В переданных основаниях нет строк по учредителям или руководителям."
    return "\n\n".join(sections)


def _num(value, default=0.0):
    """Безопасно приводит значение к числу (для логики раздела 2)."""
    try:
        number = float(value)
        return default if number != number else number
    except (TypeError, ValueError):
        return default


def _fmt_count(value):
    """6.0 -> '6' для текста раздела 2."""
    number = _num(value)
    return str(int(number)) if number.is_integer() else str(number)


def _found(row, group):
    return _value(row, group + "_candidate_found", 0) == 1


def _join_ru(items):
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " и " + items[-1]


def _section_observed(row):
    """Раздел 1: фиксированный перечень рассчитанных совпадений."""
    lines = []
    if _found(row, "staff"):
        lines.append(
            "- **Сотрудники**: общих сотрудников — {0}, доля сотрудников исходной "
            "организации — {1}%, регулярных совпадений — {2}, последовательных "
            "периодов — {3}.".format(
                _value(row, "staff_common_count"),
                _value(row, "staff_source_share_pct"),
                _value(row, "staff_regular_common_count"),
                _value(row, "staff_sequential_count")))
    for group, title in (("suppliers", "Поставщики"), ("buyers", "Покупатели")):
        if _found(row, group):
            lines.append(
                "- **{0}**: общих контрагентов — {1}, доля контрагентов исходной — {2}%, "
                "покрытие оборота исходной — {3}%, оборот исходной с общими — {4}.".format(
                    title,
                    _value(row, group + "_common_count"),
                    _value(row, group + "_source_count_share_pct"),
                    _value(row, group + "_source_turnover_share_pct"),
                    _value(row, group + "_common_source_amount")))
    if _found(row, "kkt"):
        lines.append(
            "- **ККТ**: общих записей — {0}, доля ККТ исходной организации — {1}%.".format(
                _value(row, "kkt_common_count"),
                _value(row, "kkt_source_share_pct")))
    for group, title in (("founders", "Учредители"), ("directors", "Руководители")):
        if _found(row, group):
            lines.append(
                "- **{0}**: совпавших физлиц — {1}, оснований — {2}.".format(
                    title,
                    _value(row, group + "_common_persons"),
                    _value(row, group + "_basis_count")))
    if not lines:
        lines.append("- Совпадений среди успешно проверенных направлений не найдено.")
    return "**1. Наблюдаемые совпадения**\n\n" + "\n".join(lines)


def _section_assessment(row):
    """Раздел 2: короткая связная оценка без повторения цифр из раздела 1."""
    names = [
        label for group, label in (
            ("staff", "сотрудникам"), ("suppliers", "поставщикам"),
            ("buyers", "покупателям"), ("kkt", "ККТ"),
            ("founders", "учредителям"), ("directors", "руководителям"),
        ) if _found(row, group)
    ]
    title = "**2. Осторожная оценка согласованности данных**\n\n"
    if not names:
        return title + (
            "Совпадений среди успешно проверенных направлений не выявлено. "
            "Это не исключает связь: направления со статусом «Нет данных» "
            "или «Ошибка» не проверялись.")

    parts = [f"Данные показывают наличие совпадений по {_join_ru(names)}."]

    if _found(row, "staff"):
        regular = _num(_value(row, "staff_regular_common_count", 0))
        sequential = _num(_value(row, "staff_sequential_count", 0))
        if regular <= 0 and sequential <= 0:
            parts.append(
                "Однако совпадения не демонстрируют регулярность или "
                "последовательность, что ограничивает выводы о переходе бизнеса.")
        elif regular <= 0:
            parts.append(
                "Регулярных совпадений по сотрудникам не выявлено, хотя "
                f"последовательные периоды зафиксированы ({_fmt_count(sequential)}), "
                "поэтому выводы о переходе бизнеса остаются ограниченными.")
        elif sequential <= 0:
            parts.append(
                "Регулярные совпадения по сотрудникам есть, но последовательность "
                "периодов не подтверждена, что ограничивает выводы о переходе бизнеса.")
        else:
            parts.append(
                "Совпадения по сотрудникам регулярны и последовательны во времени, "
                "это повышает согласованность данных, но не подтверждает переход бизнеса.")
    else:
        parts.append(
            "Совпадений по сотрудникам нет, поэтому регулярность и "
            "последовательность оценить нельзя.")

    parts.append("Всё наблюдаемое — это совпадения, а не доказательства.")
    return title + " ".join(parts)


def build_fixed_sections(candidate_row):
    """Разделы 1–2 формируются кодом, чтобы модель их не переписывала."""
    return _section_observed(candidate_row) + "\n\n" + _section_assessment(candidate_row)


def build_prompt(source_inn, candidate_inn, candidate_row, warnings,
                 states=None, evidence_by_group=None):
    fixed_sections = build_fixed_sections(candidate_row)
    role_text = _role_evidence_text(evidence_by_group)
    warning_text = "\n".join(f"- {item}" for item in (warnings or [])) \
        or "- Отдельные ограничения не переданы."

    return f"""Ты — беспристрастный аналитик. Продолжи аналитическое заключение только по переданным расчётам.

ЖЁСТКИЕ ПРАВИЛА:
1. Не придумывай суммы, даты, названия, события, роли, ФИО, ИНН или причины, которых нет во входных данных.
2. Не утверждай виновность, умысел, незаконность или доказанный факт перехода бизнеса.
3. Если направление имеет статус «Нет данных» или «Ошибка», не трактуй его как отсутствие совпадений.
4. Не называй индексом или вероятностью необученный экспертный вывод.
5. Разделы 1 и 2 уже написаны (см. ниже). НЕ повторяй и НЕ пересказывай их, не перечисляй снова цифры.
6. Совпадения по учредителям и руководителям есть, если они указаны в разделе 1. Не пиши, что оснований нет.
7. Пиши по-русски. Начни ответ строго с заголовка «**3. Возможные добросовестные объяснения**».

ОБЪЕКТЫ АНАЛИЗА:
- Исходная организация: ИНН {source_inn}
- Кандидат: ИНН {candidate_inn}

СТАТУСЫ НАПРАВЛЕНИЙ:
{_state_text(states)}

УЖЕ НАПИСАННЫЕ РАЗДЕЛЫ 1–2 (только для контекста):
{fixed_sections}

ОСНОВАНИЯ ПО УЧРЕДИТЕЛЯМ И РУКОВОДИТЕЛЯМ (для контекста):
{role_text}

ПРИМЕРЫ СТРОК ВСЕХ ОСНОВАНИЙ:
{_evidence_text(evidence_by_group)}

ОГРАНИЧЕНИЯ И ПРЕДУПРЕЖДЕНИЯ:
{warning_text}

НАПИШИ ТОЛЬКО РАЗДЕЛЫ:
**3. Возможные добросовестные объяснения** — только как гипотезы.
**4. Что дополнительно проверить или запросить.**
**5. Ограничения вывода.**
"""

def generate_conclusion(source_inn, candidate_inn, candidate_row, warnings,
                        states=None, evidence_by_group=None,
                        model_name=DEFAULT_MODEL):
    if not is_ollama_online():
        raise RuntimeError(
            "Ollama не обнаружена по адресу " + OLLAMA_ROOT + ".\n\n"
            "Откройте приложение Ollama или запустите ollama serve, "
            "если сервер ещё не работает.\n"
            "Проверьте установленные модели: ollama list\n"
            "При необходимости загрузите: ollama pull " + model_name
        )

    prompt = build_prompt(
        source_inn,
        candidate_inn,
        candidate_row,
        warnings,
        states,
        evidence_by_group,
    )

    # Дополнительная инструкция для Qwen3.
    # Основное отключение режима рассуждений — параметр think ниже.
    prompt += "\n\nДай сразу итоговое заключение на русском языке.\n/no_think"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 2048,
        },
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            raw_response = response.read().decode("utf-8")

    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(
            "utf-8", errors="replace"
        )[:1500]

        raise RuntimeError(
            f"Ollama вернула HTTP {exc.code}: {detail}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Не удалось обратиться к Ollama: {exc.reason}"
        ) from exc

    except TimeoutError as exc:
        raise RuntimeError(
            "Истекло время ожидания ответа Ollama — 300 секунд. "
            "Модель могла не успеть загрузиться или завершить генерацию. "
            "Проверьте её работу командой: ollama run " + model_name
        ) from exc

    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Ollama вернула некорректный JSON. "
            "Проверьте адрес API и параметр stream=False."
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError(
            "Неожиданный формат ответа Ollama: ожидался JSON-объект."
        )

    if result.get("error"):
        raise RuntimeError(
            f"Ошибка Ollama: {result['error']}"
        )

    # Для /api/generate итоговый текст находится в поле response.
    response_text = result.get("response")
    text = (
        response_text.strip()
        if isinstance(response_text, str)
        else ""
    )

    if not text:
        thinking = result.get("thinking")
        thinking_chars = (
            len(thinking) if isinstance(thinking, str) else 0
        )

        details = (
            f"Модель: {result.get('model', model_name)}\n"
            f"Генерация завершена: {result.get('done', 'не указано')}\n"
            f"Причина завершения: "
            f"{result.get('done_reason', 'не указана')}\n"
            f"Сгенерировано токенов: "
            f"{result.get('eval_count', 'не указано')}\n"
            f"Символов в поле thinking: {thinking_chars}"
        )

        if result.get("done_reason") == "length":
            hint = (
                "Достигнут лимит генерации. "
                "Проверьте, отключён ли режим рассуждений; "
                "при необходимости увеличьте num_predict."
            )
        elif thinking_chars:
            hint = (
                "Получено поле рассуждений, но нет итогового текста. "
                "Проверьте, поддерживает ли установленная версия "
                "Ollama отключение рассуждений для этой модели."
            )
        else:
            hint = (
                "Проверьте генерацию напрямую командой: "
                "ollama run " + model_name
            )

        raise RuntimeError(
            "Ollama вернула пустое заключение.\n\n"
            + details
            + "\n\n"
            + hint
        )

# Если модель всё же повторила разделы 1–2, отрезаем всё до раздела 3.
    match = re.search(r"(^|\n)\s*(\*\*|#+\s*)?3\.", text)
    if match:
        text = text[match.start():].lstrip()

    return build_fixed_sections(candidate_row) + "\n\n" + text