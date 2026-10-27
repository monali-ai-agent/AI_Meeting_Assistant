"""Send email through SMTP (Gmail: use an App Password, host smtp.gmail.com, port 465 or 587)."""
import smtplib
import ssl
from email.message import EmailMessage


class Emailer:
    def __init__(self, host="smtp.gmail.com", port=465, user="", password="", my_email=""):
        self.host, self.port, self.user, self.password = host, int(port), user, password
        self.my_email = my_email or user

    @property
    def configured(self):
        return bool(self.host and self.user and self.password)

    def send(self, to, subject, body):
        if not self.configured:
            raise RuntimeError("Email is not configured (set SMTP user and app password in the sidebar).")
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = self.user, to, subject
        msg.set_content(body)
        if self.port == 465:
            with smtplib.SMTP_SSL(self.host, self.port, context=ssl.create_default_context(), timeout=20) as s:
                s.login(self.user, self.password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(self.host, self.port, timeout=20) as s:
                s.starttls(context=ssl.create_default_context())
                s.login(self.user, self.password)
                s.send_message(msg)
        return {"sent": True, "to": to, "subject": subject}
