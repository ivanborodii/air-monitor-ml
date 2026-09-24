#!/usr/bin/env python3
"""Alert if raw_observations has gone stale in MotherDuck.

Runs on GitHub Actions (not on the Pi), so it can detect the Raspberry Pi
being unexpectedly off, not just the pipeline process crashing. Checks
MotherDuck's raw_observations for a row within the last GAP_HOURS hours;
if none is found (or the query itself fails), emails an alert -- but only
during ALERT_WINDOW local hours in Europe/Kyiv, so a Pi that has been off
since last night doesn't spam an inbox at 3am.

Env vars (see .github/workflows/data-gap-alert.yml):
  MOTHERDUCK_TOKEN    - MotherDuck PAT (repo secret)
  GMAIL_APP_PASSWORD  - Gmail app password for GMAIL_USER (repo secret)
  GMAIL_USER          - sending/receiving Gmail address (workflow env, not secret)
"""

import os
import smtplib
import sys
import traceback
from datetime import datetime, timezone
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

MD_DB = "sensors_data"
MD_TABLE = "raw_observations"
GAP_HOURS = 6
ALERT_WINDOW = (9, 21)  # [start, end) local hour, Europe/Kyiv
ALERT_TZ = ZoneInfo("Europe/Kyiv")

GMAIL_USER = os.environ.get("GMAIL_USER", "ivanborodii@gmail.com")
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
MOTHERDUCK_TOKEN = os.environ["MOTHERDUCK_TOKEN"]


def send_email(subject: str, body: str) -> None:
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = GMAIL_USER
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        smtp.send_message(msg)


def main() -> int:
    now_utc = datetime.now(timezone.utc)
    now_kyiv = now_utc.astimezone(ALERT_TZ)

    if os.environ.get("FORCE_TEST_ALERT", "").lower() in ("1", "true"):
        send_email(
            "[air-monitor] Test alert (forced manually)",
            "This is a manually forced test email from check_data_gap.py "
            "(FORCE_TEST_ALERT=true), not a real data-gap detection. If "
            "you got this, SMTP delivery from GitHub Actions works.\n\n"
            f"Sent at {now_utc.isoformat()} UTC / {now_kyiv:%Y-%m-%d %H:%M} Kyiv.",
        )
        print("FORCE_TEST_ALERT set; sent test email and exiting.")
        return 0

    if not (ALERT_WINDOW[0] <= now_kyiv.hour < ALERT_WINDOW[1]):
        print(f"Outside alert window ({now_kyiv:%H:%M} Kyiv) - skipping check.")
        return 0

    try:
        import duckdb

        con = duckdb.connect(f"md:{MD_DB}?motherduck_token={MOTHERDUCK_TOKEN}")
        row = con.execute(f"SELECT max(measured_at) FROM {MD_TABLE}").fetchone()
        last_ts = row[0] if row else None
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        send_email(
            "[air-monitor] Data-gap check itself failed",
            "The check script raised an exception before it could determine "
            "freshness. Traceback:\n\n" + tb,
        )
        print("Check failed with an exception; sent failure alert.", file=sys.stderr)
        return 1

    if last_ts is None:
        send_email(
            "[air-monitor] No sensor data found at all",
            f"raw_observations in MotherDuck ({MD_DB}) returned no rows.\n"
            f"Checked at {now_utc.isoformat()} UTC / {now_kyiv:%Y-%m-%d %H:%M} Kyiv.",
        )
        print("No rows found; sent alert.")
        return 0

    age = now_utc - last_ts
    age_hours = age.total_seconds() / 3600
    if age_hours > GAP_HOURS:
        last_ts_kyiv = last_ts.astimezone(ALERT_TZ)
        send_email(
            "[air-monitor] No new sensor data in over 6 hours",
            "air-monitor's raw_observations table has not received a new row "
            f"in {age_hours:.1f} hours.\n\n"
            f"Last known reading: {last_ts.isoformat()} UTC "
            f"({last_ts_kyiv:%Y-%m-%d %H:%M} Kyiv)\n"
            f"Checked at:         {now_utc.isoformat()} UTC "
            f"({now_kyiv:%Y-%m-%d %H:%M} Kyiv)\n\n"
            "This usually means the Raspberry Pi lost power or the "
            "air-monitor.service pipeline died. Check the device.",
        )
        print(f"Data gap of {age_hours:.1f}h detected; sent alert.")
    else:
        print(f"Data fresh: last row {age_hours:.2f}h ago ({last_ts.isoformat()} UTC).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
