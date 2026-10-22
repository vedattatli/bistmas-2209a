"""E-mail delivery of the monthly report (SMTP with STARTTLS; credentials from the environment / .env)."""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

from bistmas.shared.config import REPO_ROOT
from bistmas.shared.log import get_logger

log = get_logger("reporting.email")


def _env(key: str) -> str | None:
    if key in os.environ:
        return os.environ[key]
    dotenv = REPO_ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def send_report(
    subject: str,
    markdown: str,
    to: str,
    host: str,
    port: int,
    dry_run: bool = False,
    attachments: list[Path] | None = None,
) -> str:
    """Send the report; with ``dry_run`` (or missing credentials) the message is only rendered to a file."""
    user, password = _env("SMTP_USER"), _env("SMTP_PASSWORD")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user or "bistmas@localhost"
    msg["To"] = to
    msg.set_content(markdown)
    for path in attachments or []:
        msg.add_attachment(Path(path).read_bytes(), maintype="application", subtype="json", filename=Path(path).name)
    if dry_run or not (user and password and to):
        reason = "dry run" if dry_run else "SMTP_USER/SMTP_PASSWORD/email_to not configured"
        log.info("e-mail not sent (%s); message rendered only", reason)
        return f"not sent ({reason})"
    with smtplib.SMTP(host, port, timeout=60) as s:
        s.starttls()
        s.login(user, password)
        s.send_message(msg)
    log.info("report e-mailed to %s", to)
    return "sent"
