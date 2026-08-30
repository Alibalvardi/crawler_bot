import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:

    def __init__(self, path: str | None = None) -> None:
        default_path = Path(__file__).with_name("data") / "bot.sqlite3"
        self.path = Path(path or os.getenv("DATABASE_PATH", str(default_path)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                -- Remove legacy conversation persistence from older versions.
                DROP TABLE IF EXISTS messages;
                DROP TABLE IF EXISTS sessions;

                CREATE TABLE IF NOT EXISTS users (
                    telegram_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_settings (
                    telegram_id INTEGER PRIMARY KEY,
                    embedding_model TEXT NOT NULL,
                    generation_model TEXT NOT NULL,
                    crawler_depth INTEGER NOT NULL DEFAULT 2,
                    crawler_pages INTEGER NOT NULL DEFAULT 5,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (telegram_id) REFERENCES users(telegram_id)
                        ON DELETE CASCADE
                );
                """
            )

    def upsert_user(self, telegram_user: Any) -> None:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO users
                    (telegram_id, username, first_name, last_name,
                     created_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(telegram_id) DO UPDATE SET
                    username = excluded.username,
                    first_name = excluded.first_name,
                    last_name = excluded.last_name,
                    last_seen_at = excluded.last_seen_at
                """,
                (
                    telegram_user.id,
                    telegram_user.username,
                    telegram_user.first_name,
                    telegram_user.last_name,
                    now,
                    now,
                ),
            )

    def get_settings(
        self, telegram_id: int, defaults: dict[str, Any]
    ) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM user_settings WHERE telegram_id = ?",
                (telegram_id,),
            ).fetchone()
            if row is None:
                now = utc_now()
                connection.execute(
                    """
                    INSERT INTO user_settings
                        (telegram_id, embedding_model, generation_model,
                         crawler_depth, crawler_pages, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        telegram_id,
                        defaults["embedding_model"],
                        defaults["generation_model"],
                        defaults["crawler_depth"],
                        defaults["crawler_pages"],
                        now,
                    ),
                )
                return dict(defaults)

            return {
                "embedding_model": row["embedding_model"],
                "generation_model": row["generation_model"],
                "crawler_depth": row["crawler_depth"],
                "crawler_pages": row["crawler_pages"],
            }

    def update_setting(self, telegram_id: int, key: str, value: Any) -> None:
        allowed = {
            "embedding_model",
            "generation_model",
            "crawler_depth",
            "crawler_pages",
        }
        if key not in allowed:
            raise ValueError(f"Unsupported setting: {key}")

        with self._connect() as connection:
            connection.execute(
                f"UPDATE user_settings SET {key} = ?, updated_at = ? "
                "WHERE telegram_id = ?",
                (value, utc_now(), telegram_id),
            )
