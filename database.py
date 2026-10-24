"""Работа с базой данных: пользователи, автомобили, история диагностик.

Используем SQLite — он не требует отдельного сервера и идеально подходит
для первого шага. Позже, при росте нагрузки, переедем на PostgreSQL.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime

from werkzeug.security import check_password_hash, generate_password_hash

# Файл базы лежит рядом с проектом (в облаке можно переопределить через переменную)
DB_FILE = os.environ.get("OBD_DB_FILE", os.path.join(os.path.dirname(__file__), "obd.db"))


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                make TEXT,
                model TEXT,
                year TEXT,
                vin TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS diagnostics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                vehicle_id INTEGER,
                codes TEXT,
                live TEXT,
                advice TEXT,
                part TEXT,
                cost TEXT,
                health TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )


# ---------- Пользователи ----------

def create_user(email: str, password: str) -> int:
    password_hash = generate_password_hash(password)
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email, password_hash, datetime.now().isoformat(timespec="seconds")),
        )
        return cursor.lastrowid


def get_user_by_email(email: str):
    with get_connection() as conn:
        return conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()


def get_user_by_id(user_id: int):
    with get_connection() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def verify_user(email: str, password: str):
    user = get_user_by_email(email)
    if user is None or not check_password_hash(user["password_hash"], password):
        return None
    return user


# ---------- Автомобили ----------

def save_vehicle(user_id: int, data: dict) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO vehicles (user_id, make, model, year, vin, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                user_id,
                data.get("make", ""),
                data.get("model", ""),
                data.get("year", ""),
                data.get("vin", ""),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        return cursor.lastrowid


def get_vehicles(user_id: int) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM vehicles WHERE user_id = ? ORDER BY id DESC", (user_id,)
        ).fetchall()
    return [dict(row) for row in rows]


# ---------- История диагностик ----------

def save_diagnostic(user_id: int, data: dict) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO diagnostics "
            "(user_id, vehicle_id, codes, live, advice, part, cost, health, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                data.get("vehicle_id"),
                json.dumps(data.get("codes", []), ensure_ascii=False),
                json.dumps(data.get("live", {}), ensure_ascii=False),
                data.get("advice", ""),
                data.get("part", ""),
                data.get("cost", ""),
                json.dumps(data.get("health", {}), ensure_ascii=False),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        return cursor.lastrowid


def get_history(user_id: int, limit: int = 50) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM diagnostics WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()

    history = []
    for row in rows:
        item = dict(row)
        item["codes"] = json.loads(item["codes"] or "[]")
        item["live"] = json.loads(item["live"] or "{}")
        item["health"] = json.loads(item["health"] or "{}")
        history.append(item)
    return history
