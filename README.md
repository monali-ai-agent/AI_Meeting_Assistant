# 🎙️ AI Meeting Assistant – Groq-powered agents, voice + chat, email

Every agent calls the Groq API (default model `openai/gpt-oss-20b`); voice input uses Groq Whisper (`whisper-large-v3-turbo`).
No offline/rule-based mode: the app will not start without a Groq key.

## Setup
```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...                 # create at https://console.groq.com/keys (or paste in sidebar)
export MY_EMAIL=you@gmail.com
export SMTP_USER=you@gmail.com
export SMTP_PASSWORD=your-16-char-app-password   # Gmail: Google Account > Security > 2-Step Verification > App passwords
streamlit run app.py
```
(Windows PowerShell: `$env:GROQ_API_KEY="..."`.) Everything can also be typed into the sidebar.

## Email
* Gmail: `smtp.gmail.com`, port `465`, **App Password** (not your normal password). Outlook: `smtp.office365.com`, port `587`.
* Put each teammate's email in the sidebar team directory (`name, email, role, skills`).
* Send via sidebar auto-options, buttons in *Process Meeting*, or ask the agent: "email me the last summary", "remind Rahul about his tasks".
* The agent can only email you and people in the team directory.

## Notes
* Change model with `GROQ_MODEL` or the sidebar. Check https://console.groq.com/docs/models for current names; the model must support tool calling.
* Voice replies use gTTS (online) or the browser voice (offline).

## Files
`app.py` UI · `agents.py` 4 agents + tool-using chat agent · `llm.py` Groq client + tool loop · `emailer.py` SMTP · `memory.py` SQLite · `tools.py` · `voice.py`
