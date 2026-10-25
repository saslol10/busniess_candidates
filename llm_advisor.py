"""Локальные аналитические заключения через Ollama API.

Модуль не рассчитывает признаки и не устанавливает факты. Он передаёт
в локальную модель только уже рассчитанные приложением данные.
"""
import json
import os
import urllib.error
import urllib.request

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


def build_prompt(source_inn, candidate_inn, candidate_row, warnings,
                 states=None, evidence_by_group=None):
    staff_found = _value(candidate_row, "staff_candidate_found", 0) == 1
    suppliers_found = _value(candidate_row, "suppliers_candidate_found", 0) == 1
    buyers_found = _value(candidate_row, "buyers_candidate_found", 0) == 1
    kkt_found = _value(candidate_row, "kkt_candidate_found", 0) == 1
    channels = []
    if staff_found:
        channels.append(
            "- Сотрудники: общих — {0}; доля сотрудников исходной организации — {1}%; "
            "регулярных совпадений — {2}; последовательных периодов — {3}.".format(
                _value(candidate_row, "staff_common_count"),
                _value(candidate_row, "staff_source_share_pct"),
                _value(candidate_row, "staff_regular_common_count"),
                _value(candidate_row, "staff_sequential_count")))
    if suppliers_found:
        channels.append(
            "- Поставщики: общих — {0}; доля контрагентов исходной — {1}%; "
            "покрытие оборота исходной — {2}%; оборот исходной с общими — {3}.".format(
                _value(candidate_row, "suppliers_common_count"),
                _value(candidate_row, "suppliers_source_count_share_pct"),
                _value(candidate_row, "suppliers_source_turnover_share_pct"),
                _value(candidate_row, "suppliers_common_source_amount")))
    if buyers_found:
        channels.append(
            "- Покупатели: общих — {0}; доля контрагентов исходной — {1}%; "
            "покрытие оборота исходной — {2}%; оборот исходной с общими — {3}.".format(
                _value(candidate_row, "buyers_common_count"),
                _value(candidate_row, "buyers_source_count_share_pct"),
                _value(candidate_row, "buyers_source_turnover_share_pct"),
                _value(candidate_row, "buyers_common_source_amount")))
    if kkt_found:
        channels.append(
            "- ККТ: общих — {0}; доля ККТ исходной организации — {1}%.".format(
                _value(candidate_row, "kkt_common_count"),
                _value(candidate_row, "kkt_source_share_pct")))
    channels_text = "\n".join(channels) if channels else "- Совпадений среди успешно проверенных направлений не найдено."
    warning_text = "\n".join(f"- {item}" for item in (warnings or [])) or "- Отдельные ограничения не переданы."

    return f"""Ты — беспристрастный аналитик. Составь краткое аналитическое заключение только по переданным расчётам.

ЖЁСТКИЕ ПРАВИЛА:
1. Не придумывай суммы, даты, названия, события и причины, которых нет во входных данных.
2. Не утверждай виновность, умысел, незаконность или факт перехода бизнеса.
3. Разделяй наблюдаемые совпадения, гипотезы и ограничения данных.
4. Если направление имеет статус «Нет данных» или «Ошибка», не трактуй его как отсутствие совпадений.
5. Не называй индексом или вероятностью необученный экспертный вывод: в этом запросе такого покзателя нет.
6. Пиши по-русски, с заголовками и без лишней риторики.

ОБЪЕКТЫ АНАЛИЗА:
- Исходная организация: ИНН {source_inn}
- Кандидат: ИНН {candidate_inn}

СТАТУСЫ НАПРАВЛЕНИЙ:
{_state_text(states)}

РАССЧИТАННЫЕ СОВПАДЕНИЯ:
{channels_text}

ПРИМЕРЫ СТРОК ОСНОВАНИЙ:
{_evidence_text(evidence_by_group)}

ОГРАНИЧЕНИЯ И ПРЕДУПРЕЖДЕНИЯ:
{warning_text}

ФОРМАТ ОТВЕТА:
1. Наблюдаемые совпадения.
2. Осторожная оценка согласованности данных — без вывода о нарушении.
3. Возможные добросовестные объяснения, только как гипотезы.
4. Что дополнительно проверить или запросить для проверки гипотез.
5. Ограничения вывода.
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

    return text