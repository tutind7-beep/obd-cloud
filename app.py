"""OBD-Диагност — облачный бэкенд (Фаза 1).

Что умеет:
  * регистрация и вход (пароли хранятся в зашифрованном виде);
  * сохранение автомобилей пользователя;
  * советы ИИ по кодам и параметрам (с сохранением в историю);
  * диагностика по симптомам (с сохранением в историю);
  * история диагностик;
  * демо-режим параметров (реальные данные с адаптера добавятся
    в Фазе 2, когда их будет присылать мобильное приложение).

Запуск:
    python app.py            # локально
    gunicorn app:app         # на хостинге (Render)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from io import BytesIO

from flask import Flask, jsonify, render_template, request, send_file
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

import database
from diagnostics import build_hints, normalize_codes
from obd_service import DemoSource

app = Flask(__name__)

# ---------- Секретный ключ для подписи токенов ----------

SECRET_FILE = os.path.join(os.path.dirname(__file__), ".secret_key")


def _load_secret_key() -> str:
    key = os.environ.get("SECRET_KEY")
    if key:
        return key
    if os.path.isfile(SECRET_FILE):
        return open(SECRET_FILE, encoding="utf-8").read().strip()
    key = os.urandom(32).hex()
    with open(SECRET_FILE, "w", encoding="utf-8") as handle:
        handle.write(key)
    return key


SECRET_KEY = _load_secret_key()
serializer = URLSafeTimedSerializer(SECRET_KEY)

TOKEN_MAX_AGE = 60 * 60 * 24 * 30  # 30 дней

# ---------- Источник данных (в облаке — демо, адаптера нет) ----------

source = DemoSource("lean")


def current_source():
    return source


# ---------- Нейросеть DeepSeek ----------

DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

PARAM_LABELS = {
    "rpm": "обороты (об/мин)",
    "coolant": "температура охлаждающей жидкости (°C)",
    "speed": "скорость (км/ч)",
    "voltage": "напряжение (В)",
    "fuel_trim_short": "краткосрочная коррекция топлива (%)",
    "fuel_trim_long": "долгосрочная коррекция топлива (%)",
    "throttle": "положение дросселя (%)",
    "maf": "расход воздуха (г/с)",
    "intake_temp": "температура впускного воздуха (°C)",
}

SYMPTOM_CHOICES = [
    "троит двигатель",
    "дёргается при разгоне",
    "не заводится",
    "перегревается",
    "горит Check Engine",
    "потеря мощности",
    "повышенный расход топлива",
    "стук или шум из двигателя",
    "запах бензина",
    "вибрация на холостых оборотах",
    "дым из выхлопной трубы",
    "нестабильные холостые обороты",
]


def _load_api_key() -> str | None:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key.strip()
    key_file = os.path.expanduser("~/.jarvis/api_key")
    if os.path.isfile(key_file):
        try:
            return open(key_file, encoding="utf-8").read().strip() or None
        except OSError:
            return None
    return None


def _ask_deepseek(messages: list[dict]) -> tuple[str | None, str | None]:
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.4,
        "stream": False,
    }
    request_obj = urllib.request.Request(
        f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_load_api_key()}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request_obj, timeout=90) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return None, f"Ошибка нейросети (HTTP {error.code}). Проверьте ключ DeepSeek."
    except urllib.error.URLError as error:
        return None, f"Нет связи с нейросетью: {error.reason}."
    except TimeoutError:
        return None, "Нейросеть не ответила за 90 секунд."

    try:
        data = json.loads(raw)
        answer = data["choices"][0]["message"]["content"].strip()
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError):
        return None, f"Неожиданный ответ нейросети: {raw[:300]}"

    if not answer:
        return None, "Нейросеть вернула пустой ответ."
    return answer, None


def _make_summary(codes: list[str], live: dict, vehicle: dict, symptoms: str) -> str:
    parts = []
    for key, label in PARAM_LABELS.items():
        value = live.get(key)
        if value is None:
            continue
        parts.append(f"{label}: {value}")

    if not parts:
        parts.append("живые параметры пока недоступны")

    vehicle_text = ", ".join(
        value for value in (vehicle.get("make"), vehicle.get("model"), vehicle.get("year"))
        if value
    ) or "марка и модель не указаны"

    codes_text = ", ".join(codes) if codes else "ошибок нет"
    summary = (
        f"Автомобиль: {vehicle_text}. "
        "Показания: " + "; ".join(parts) + ". "
        f"Коды ошибок: {codes_text}."
    )
    if symptoms:
        summary += f" Жалоба водителя: {symptoms}."
    return summary


def _build_part_links(part: str, vehicle: dict) -> list[dict]:
    query_parts = [part] + [
        vehicle.get("make"),
        vehicle.get("model"),
        vehicle.get("year"),
    ]
    query = " ".join(value for value in query_parts if value).strip()
    if not query:
        return []

    encoded = urllib.parse.quote(query)
    return [
        {"label": "Autodoc", "url": f"https://www.autodoc.ru/search?text={encoded}"},
        {"label": "Emex", "url": f"https://emex.ru/products/search?query={encoded}"},
        {"label": "Ozon", "url": f"https://www.ozon.ru/search/?text={encoded}"},
        {"label": "Яндекс.Маркет", "url": f"https://market.yandex.ru/search?text={encoded}"},
        {"label": "Avito", "url": f"https://www.avito.ru/all?q={encoded}"},
        {"label": "Google", "url": f"https://www.google.com/search?q={encoded}"},
    ]


def _parse_json_answer(answer: str) -> dict:
    """Достаёт JSON из ответа нейросети, терпимо к обёрткам ```json ...```."""
    text = answer.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}


def _health_score(hints) -> dict:
    score = 100
    for hint in hints:
        if hint.severity == "СТОП":
            score -= 25
        elif hint.severity == "СКОРО В СЕРВИС":
            score -= 15
        else:
            score -= 8
    score = max(0, min(100, score))

    if score >= 90:
        label = "Отлично"
    elif score >= 70:
        label = "Хорошо"
    elif score >= 50:
        label = "Есть замечания"
    else:
        label = "Нужен сервис"

    return {"score": score, "label": label}


# ---------- Авторизация ----------

def current_user():
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    token = header[len("Bearer "):]
    try:
        user_id = serializer.loads(token, max_age=TOKEN_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None
    return database.get_user_by_id(user_id)


def require_auth():
    user = current_user()
    if user is None:
        return None, jsonify({"error": "Требуется вход. Передайте токен в заголовке Authorization."}), 401
    return user, None, None


# ---------- Публичные страницы ----------

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def api_health():
    return jsonify({"ok": True})


# ---------- Регистрация и вход ----------

@app.post("/api/register")
def api_register():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if "@" not in email or len(password) < 6:
        return jsonify({"error": "Укажите корректный email и пароль от 6 символов."}), 400
    if database.get_user_by_email(email):
        return jsonify({"error": "Такой email уже зарегистрирован."}), 409

    user_id = database.create_user(email, password)
    token = serializer.dumps(user_id)
    return jsonify({"token": token, "email": email}), 201


@app.post("/api/login")
def api_login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = database.verify_user(email, password)
    if user is None:
        return jsonify({"error": "Неверный email или пароль."}), 401

    token = serializer.dumps(user["id"])
    return jsonify({"token": token, "email": user["email"]})


@app.get("/api/me")
def api_me():
    user, error_response, status = require_auth()
    if user is None:
        return error_response, status
    return jsonify(
        {
            "email": user["email"],
            "vehicles": database.get_vehicles(user["id"]),
        }
    )


# ---------- Автомобили ----------

@app.post("/api/vehicles")
def api_vehicles():
    user, error_response, status = require_auth()
    if user is None:
        return error_response, status

    data = request.get_json(silent=True) or {}
    vehicle_id = database.save_vehicle(user["id"], data)
    return jsonify({"vehicle_id": vehicle_id}), 201


# ---------- Демо-данные (в облаке адаптера нет) ----------

@app.get("/api/live")
def api_live():
    active = current_source()
    codes, live = active.read()
    codes = normalize_codes(codes)
    hints = build_hints(codes, live)

    return jsonify(
        {
            "source": "demo",
            "connected": True,
            "error": None,
            "codes": codes,
            "live": live,
            "hints": [hint.to_dict() for hint in hints],
            "stopped": any(hint.severity == "СТОП" for hint in hints),
            "health": _health_score(hints),
            "scenario": active.scenario,
            "updated_at": datetime.now().strftime("%H:%M:%S"),
        }
    )


@app.post("/api/demo/<scenario>")
def api_set_demo(scenario: str):
    if scenario not in DemoSource.SCENARIOS:
        return jsonify({"error": "Неизвестный сценарий"}), 404
    source.scenario = scenario
    return jsonify({"scenario": scenario})


# ---------- Советы ИИ (с сохранением в историю) ----------

@app.post("/api/advice")
def api_advice():
    user = current_user()  # вход необязателен: гости тоже получают совет

    if not _load_api_key():
        return jsonify({"error": "Ключ DeepSeek не настроен на сервере."}), 500

    data = request.get_json(silent=True) or {}
    vehicle = {
        "make": (data.get("make") or "").strip(),
        "model": (data.get("model") or "").strip(),
        "year": str(data.get("year") or "").strip(),
    }
    symptoms = (data.get("symptoms") or "").strip()

    codes, live = current_source().read()
    codes = normalize_codes(codes)
    hints = build_hints(codes, live)
    summary = _make_summary(codes, live, vehicle, symptoms)

    messages = [
        {
            "role": "system",
            "content": (
                "Ты — опытный автомеханик. Пользователь передаёт данные "
                "автомобиля, показания OBD-II адаптера, коды ошибок и жалобу "
                "водителя. Ответь строго в формате JSON без пояснений: "
                '{"advice": "рекомендация на русском языке, 3–6 предложений", '
                '"part": "название запчасти, которая скорее всего вышла из '
                'строя, или пустая строка", '
                '"cost": "примерная стоимость ремонта в рублях, например '
                '\'3000–7000 ₽\', или пустая строка"}.'
            ),
        },
        {"role": "user", "content": summary},
    ]

    answer, error = _ask_deepseek(messages)
    if error:
        return jsonify({"error": error}), 502

    parsed = _parse_json_answer(answer)
    advice = str(parsed.get("advice") or "").strip() or answer
    part = str(parsed.get("part") or "").strip()
    cost = str(parsed.get("cost") or "").strip()

    if user is not None:
        database.save_diagnostic(
            user["id"],
            {
                "codes": codes,
                "live": live,
                "advice": advice,
                "part": part,
                "cost": cost,
                "health": _health_score(hints),
            },
        )

    return jsonify(
        {
            "advice": advice,
            "part": part,
            "cost": cost,
            "links": _build_part_links(part, vehicle),
            "error": None,
            "updated_at": datetime.now().strftime("%H:%M:%S"),
        }
    )


# ---------- Диагностика по симптомам ----------

@app.post("/api/symptoms")
def api_symptoms():
    user = current_user()  # вход необязателен

    if not _load_api_key():
        return jsonify({"error": "Ключ DeepSeek не настроен на сервере."}), 500

    data = request.get_json(silent=True) or {}
    selected = [
        symptom for symptom in (data.get("symptoms") or [])
        if isinstance(symptom, str) and symptom.strip()
    ]
    if not selected:
        return jsonify({"error": "Выберите хотя бы один симптом."}), 400

    vehicle = {
        "make": (data.get("make") or "").strip(),
        "model": (data.get("model") or "").strip(),
        "year": str(data.get("year") or "").strip(),
    }
    vehicle_text = ", ".join(
        value for value in (vehicle.get("make"), vehicle.get("model"), vehicle.get("year"))
        if value
    ) or "марка и модель не указаны"

    summary = f"Автомобиль: {vehicle_text}. Симптомы: {', '.join(selected)}."

    messages = [
        {
            "role": "system",
            "content": (
                "Ты — опытный диагност. По списку симптомов и данным автомобиля "
                "предположи вероятные причины. Ответь строго в формате JSON: "
                '{"diagnosis": "разбор на русском: 3–5 вероятных причин с '
                'пояснением", "parts": ["запчасть 1", "запчасть 2"]}.'
            ),
        },
        {"role": "user", "content": summary},
    ]

    answer, error = _ask_deepseek(messages)
    if error:
        return jsonify({"error": error}), 502

    parsed = _parse_json_answer(answer)
    diagnosis = str(parsed.get("diagnosis") or "").strip() or answer
    parts = [
        str(part).strip()
        for part in (parsed.get("parts") or [])
        if str(part).strip()
    ]

    if user is not None:
        database.save_diagnostic(
            user["id"],
            {
                "codes": [],
                "live": {},
                "advice": diagnosis,
                "part": ", ".join(parts),
                "cost": "",
                "health": {},
            },
        )

    return jsonify(
        {
            "diagnosis": diagnosis,
            "parts": [
                {"name": part, "links": _build_part_links(part, vehicle)}
                for part in parts
            ],
            "error": None,
            "updated_at": datetime.now().strftime("%H:%M:%S"),
        }
    )


# ---------- История ----------

@app.get("/api/history")
def api_history():
    user, error_response, status = require_auth()
    if user is None:
        return error_response, status
    return jsonify({"history": database.get_history(user["id"])})


# ---------- VIN ----------

VIN_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVin/{vin}?format=json"


@app.get("/api/vin/<vin>")
def api_vin(vin: str):
    vin = vin.strip().upper()
    if len(vin) != 17 or any(ch in "IOQ" for ch in vin):
        return jsonify({"vehicle": None, "error": "VIN должен состоять из 17 символов без I, O, Q."}), 400

    try:
        with urllib.request.urlopen(VIN_URL.format(vin=vin), timeout=20) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as error:
        return jsonify({"vehicle": None, "error": f"Нет связи с сервисом VIN: {error.reason}."}), 502
    except TimeoutError:
        return jsonify({"vehicle": None, "error": "Сервис VIN не ответил."}), 504

    try:
        data = json.loads(raw)
        results = data["Results"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return jsonify({"vehicle": None, "error": "Неожиданный ответ сервиса VIN."}), 502

    fields = {}
    for item in results:
        if isinstance(item, dict) and item.get("Variable"):
            fields[item["Variable"]] = (item.get("Value") or "").strip()

    make = fields.get("Make", "")
    if not make:
        return jsonify(
            {"vehicle": None, "error": "По этому VIN ничего не найдено. Введите данные вручную."}
        ), 404

    return jsonify(
        {
            "vehicle": {
                "make": make,
                "model": fields.get("Model", ""),
                "year": fields.get("Model Year", ""),
            },
            "error": None,
        }
    )


# ---------- Отчёт ----------

@app.get("/api/report")
def api_report():
    active = current_source()
    codes, live = active.read()
    codes = normalize_codes(codes)
    hints = build_hints(codes, live)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "demo",
        "codes": codes,
        "live": live,
        "hints": [hint.to_dict() for hint in hints],
        "health": _health_score(hints),
    }

    payload = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
    filename = f"obd_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    return send_file(
        BytesIO(payload),
        mimetype="application/json",
        as_attachment=True,
        download_name=filename,
    )


# ---------- Запуск ----------

if __name__ == "__main__":
    database.init_db()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
