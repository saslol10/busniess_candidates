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


def _details_frame(value):
    """
    Приводит основание к DataFrame.

    Поддерживает два формата:
    1. DataFrame напрямую;
    2. результат roles.search_roles(), где DataFrame находится
       в поле details.
    """
    if value is None:
        return None

    if hasattr(value, "empty") and hasattr(value, "head"):
        return value

    if isinstance(value, dict):
        details = value.get("details")
        if details is not None:
            return _details_frame(details)

    return None


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
    for group, value in evidence_by_group.items():
        if group in {"founders", "directors"}:
            continue

        frame = _details_frame(value)

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


def _role_evidence_text(evidence_by_group, candidate_inn=None):
    """Возвращает подробные основания по учредителям и руководителям."""
    if not evidence_by_group:
        return "- Подробные основания по учредителям и руководителям не переданы."

    sections = []

    for group, title in (
        ("founders", "Учредители"),
        ("directors", "Руководители"),
    ):
        value = evidence_by_group.get(group)
        frame = _details_frame(value)

        if frame is None or frame.empty:
            continue

        # В search_roles() возвращаются основания по всем кандидатам.
        # Оставляем только выбранного кандидата.
        if candidate_inn is not None:
            candidate_column = next(
                (
                    column
                    for column in (
                        "ИНН кандидата",
                        "candidate_inn",
                        "ИНН организации-кандидата",
                    )
                    if column in frame.columns
                ),
                None,
            )

            if candidate_column:
                frame = frame[
                    frame[candidate_column].astype(str).str.strip()
                    == str(candidate_inn).strip()
                ].copy()

        if frame.empty:
            continue

        lines = [f"**{title}**:"]

        for record in frame.head(12).to_dict("records"):
            inn = record.get("ИНН физлица", "")
            fio = record.get("ФИО", "")
            source_role = record.get("Роль в исходной организации", "")
            candidate_role = record.get("Роль в организации-кандидате", "")
            period = record.get("Период (для сотрудников)", "")
            basis = record.get("Основание", "")

            person = fio or "ФИО не указано"
            identity = f"{person}"
            if inn:
                identity += f" (ИНН физлица: {inn})"

            line = f"- {identity}"

            roles = []
            if source_role:
                roles.append(f"в исходной организации — {source_role}")
            if candidate_role:
                roles.append(f"в организации-кандидате — {candidate_role}")

            if roles:
                line += "; " + ", ".join(roles)

            if period:
                line += f"; период: {period}"

            if basis:
                line += f"; основание: {basis}"

            lines.append(line)

        if len(lines) > 1:
            sections.append("\n".join(lines))

    if not sections:
        return "- В переданных основаниях нет строк по учредителям или руководителям."

    return "\n\n".join(sections)

def build_prompt(source_inn, candidate_inn, candidate_row, warnings,
                 states=None, evidence_by_group=None):
    """Собирает промпт с фиксированным форматом раздела наблюдаемых совпадений."""
    def found(group):
        return _value(candidate_row, group + "_candidate_found", 0) == 1

    channels = []
    if found("staff"):
        channels.append(
            "- **Сотрудники**: общий совпадающий сотрудник — {0}, "
            "доля сотрудников исходной организации — {1}%, "
            "регулярных совпадений — {2}, последовательных периодов — {3}.".format(
                _value(candidate_row, "staff_common_count"),
                _value(candidate_row, "staff_source_share_pct"),
                _value(candidate_row, "staff_regular_common_count"),
                _value(candidate_row, "staff_sequential_count")))
    if found("suppliers"):
        channels.append(
            "- **Поставщики**: общий контрагент — {0}, "
            "доля контрагентов исходной — {1}%, "
            "покрытие оборота исходной — {2}%, "
            "оборот исходной с общими — {3}.".format(
                _value(candidate_row, "suppliers_common_count"),
                _value(candidate_row, "suppliers_source_count_share_pct"),
                _value(candidate_row, "suppliers_source_turnover_share_pct"),
                _value(candidate_row, "suppliers_common_source_amount")))
    if found("buyers"):
        channels.append(
            "- **Покупатели**: общий контрагент — {0}, "
            "доля контрагентов исходной — {1}%, "
            "покрытие оборота исходной — {2}%, "
            "оборот исходной с общими — {3}.".format(
                _value(candidate_row, "buyers_common_count"),
                _value(candidate_row, "buyers_source_count_share_pct"),
                _value(candidate_row, "buyers_source_turnover_share_pct"),
                _value(candidate_row, "buyers_common_source_amount")))
    if found("kkt"):
        channels.append(
            "- **ККТ**: общих записей — {0}, "
            "доля ККТ исходной организации — {1}%.".format(
                _value(candidate_row, "kkt_common_count"),
                _value(candidate_row, "kkt_source_share_pct")))

    for group, title in (("founders", "Учредители"), ("directors", "Руководители")):
        if found(group):
            channels.append(
                "- **{0}**: совпавших физлиц — {1}, оснований — {2}.".format(
                    title,
                    _value(candidate_row, group + "_common_persons"),
                    _value(candidate_row, group + "_basis_count")))

    channels_text = "\n".join(channels) if channels else (
        "- Совпадений среди успешно проверенных направлений не найдено.")
    status_text = _state_text(states)
    role_text = _role_evidence_text(
        evidence_by_group,
        candidate_inn=candidate_inn,
    )
    warning_text = "\n".join(
        f"- {item}" for item in (warnings or [])
    ) or "- Отдельные ограничения не переданы."

    return f"""Ты — беспристрастный аналитик. Составь подробное аналитическое заключение только по переданным расчётам.

ЖЁСТКИЕ ПРАВИЛА:
1. Не придумывай суммы, даты, названия, события, роли, ФИО, ИНН или причины, которых нет во входных данных.
2. Не утверждай виновность, умысел, незаконность или доказанный факт перехода бизнеса.
3. Разделяй наблюдаемые совпадения, гипотезы и ограничения данных.
4. Если направление имеет статус «Нет данных» или «Ошибка», не трактуй его как отсутствие совпадений.
5. Не называй индексом или вероятностью необученный экспертный вывод: в этом запросе такого показателя нет.
6. Пиши по-русски, с заголовками и маркированными пунктами.
7. Начинай ответ строго с заголовка **1. Наблюдаемые совпадения**.
8. В первом разделе сначала выведи статусы направлений дословно из блока «СТАТУСЫ НАПРАВЛЕНИЙ», затем все рассчитанные пункты из блока «РАССЧИТАННЫЕ СОВПАДЕНИЯ».
9. Не пересчитывай и не округляй значения. Используй числа и формулировки из расчётов.
10. Если в расчётах есть учредители или руководители, обязательно добавь их отдельными пунктами после ККТ.
11. Для каждого человека из раздела «ОСНОВАНИЯ ПО УЧРЕДИТЕЛЯМ И РУКОВОДИТЕЛЯМ» обязательно укажи ФИО, ИНН физлица, роль в исходной организации, роль в организации-кандидате, период (если передан) и основание.
12. Учредителей и руководителей опиши отдельными подразделами внутри раздела 1. Совпадение по ИНН физлица является результатом сопоставления данных, но само по себе не доказывает переход бизнеса.
13. После раздела 1 используй ровно такие разделы: «2. Осторожная оценка согласованности данных», «3. Возможные добросовестные объяснения», «4. Что дополнительно проверить или запросить», «5. Ограничения вывода».

ОБЪЕКТЫ АНАЛИЗА:
- Исходная организация: ИНН {source_inn}
- Кандидат: ИНН {candidate_inn}

СТАТУСЫ НАПРАВЛЕНИЙ:
{status_text}

РАССЧИТАННЫЕ СОВПАДЕНИЯ:
{channels_text}

ОСНОВАНИЯ ПО УЧРЕДИТЕЛЯМ И РУКОВОДИТЕЛЯМ:
{role_text}

ОГРАНИЧЕНИЯ И ПРЕДУПРЕЖДЕНИЯ:
{warning_text}

ФОРМАТ ПЕРВОГО РАЗДЕЛА:
**1. Наблюдаемые совпадения**
Статусы направлений:
[перечисли статусы из блока выше]

[перечисли пункты сотрудников, поставщиков, покупателей и ККТ из расчётов]

[если есть данные, отдельные пункты **Учредители** и **Руководители** с подробными людьми и ролями]

После этого напиши остальные разделы в указанном порядке:
2. Осторожная оценка согласованности данных.
3. Возможные добросовестные объяснения — только как гипотезы.
4. Что дополнительно проверить или запросить.
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