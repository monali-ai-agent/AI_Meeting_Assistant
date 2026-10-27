"""AI Meeting Assistant – real Groq-powered multi-agent app (text + voice chat + email).
Run:  streamlit run app.py
"""
import hashlib
import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from agents import Orchestrator
from emailer import Emailer
from llm import LLM, LLMError
from memory import Memory
from tools import DEFAULT_TEAM, DocumentParser, TaskTool
from voice import browser_speech_html, transcribe, transcribe_groq, tts_mp3

st.set_page_config(page_title="AI Meeting Assistant", page_icon="🎙️", layout="wide")
S = st.session_state
S.setdefault("messages", [{"role": "assistant", "audio": None, "tools": [], "content":
    "Hi! I'm your meeting assistant. Paste a transcript, upload a file in **Process Meeting**, or ask me anything — "
    "typing or 🎙️ voice. Try: *what's pending?* or *email me a summary of the last meeting*."}])
for k, v in {"mic_n": 0, "last_audio": None, "play_idx": None, "speak_text": None, "last_result": None, "team": DEFAULT_TEAM}.items():
    S.setdefault(k, v)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🎙️ Meeting Assistant")
    st.subheader("🔑 Groq API")
    key = st.text_input("Groq API key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    model = st.text_input("Model", value=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))

    st.subheader("📧 Email (SMTP)")
    my_email = st.text_input("Your email (summaries go here)", value=os.getenv("MY_EMAIL", ""))
    smtp_user = st.text_input("SMTP username (sender)", value=os.getenv("SMTP_USER", my_email))
    smtp_pass = st.text_input("SMTP / app password", type="password", value=os.getenv("SMTP_PASSWORD", ""))
    c1, c2 = st.columns(2)
    smtp_host = c1.text_input("Host", value=os.getenv("SMTP_HOST", "smtp.gmail.com"))
    smtp_port = c2.number_input("Port", value=int(os.getenv("SMTP_PORT", "465")), step=1)
    auto_summary = st.checkbox("Auto-email summary to me after each meeting", value=True)
    auto_owners = st.checkbox("Auto-email task owners their assignments", value=False)

    st.subheader("🔊 Voice")
    voice_on = st.toggle("Speak replies", value=True)
    engine = st.radio("Voice engine", ["gTTS (natural, online)", "Browser (instant, offline)"])
    LANGS = {"English (India)": ("en-IN", "en"), "English (US)": ("en-US", "en"), "Hindi": ("hi-IN", "hi"), "Telugu": ("te-IN", "te")}
    stt_lang, tts_lang = LANGS[st.selectbox("Language", list(LANGS))]

    st.subheader("👥 Team directory")
    team_txt = st.text_area("name, email, role, skills (one per line)", height=170, value="\n".join(
        f"{m['name']}, {m.get('email', '')}, {m['role']}, {' '.join(m['skills'])}" for m in S.team))
    team = []
    for line in team_txt.splitlines():
        p = [x.strip() for x in line.split(",", 3)]
        if len(p) == 4 and p[0]:
            team.append({"name": p[0], "email": p[1], "role": p[2], "skills": p[3].lower().split()})
    S.team = team or DEFAULT_TEAM

# ------------------------------------------------------------------ API key is mandatory
if not key:
    st.title("🎙️ AI Meeting Assistant")
    st.error("Enter your Groq API key in the sidebar (or set GROQ_API_KEY) to start. This app has no offline mode.")
    st.stop()
try:
    llm = LLM(key, model)
except LLMError as e:
    st.error(str(e)); st.stop()


@st.cache_resource
def get_memory():
    return Memory("data/meetings.db")


mem = get_memory()
tools = TaskTool(mem)
emailer = Emailer(smtp_host, smtp_port, smtp_user, smtp_pass, my_email)
orch = Orchestrator(llm, mem, tools, DocumentParser(), S.team, emailer)


def speak(text, idx):
    if not voice_on or not text:
        return None
    if engine.startswith("Browser"):
        S.speak_text = text
        return None
    audio, err = tts_mp3(text, tts_lang)
    if err:
        st.toast(err + " – using browser voice", icon="⚠️")
        S.speak_text = text
        return None
    S.play_idx = idx
    return audio


def email_after_processing(res):
    notes = []
    try:
        if auto_summary:
            notes.append(("summary", orch.email_summary(res)))
        if auto_owners:
            notes += [(o["owner"], o) for o in orch.notify_owners(res)]
    except Exception as e:
        notes.append(("email", {"sent": False, "error": str(e)}))
    return notes


def handle_user(text, via_voice=False):
    S.messages.append({"role": "user", "content": ("🎙️ " if via_voice else "") + text, "audio": None, "tools": []})
    try:
        with st.spinner("Agents working…"):
            out = orch.chat(text, S.messages[:-1])
            if out.get("result"):
                S.last_result = out["result"]
                notes = email_after_processing(out["result"])
                if notes:
                    out["reply"] += "\n\n" + "\n".join(
                        f"{'✉️ Sent' if n.get('sent') else '⚠️ Email failed'}: {k}" + (f" – {n.get('error')}" if n.get("error") else "") for k, n in notes)
    except Exception as e:
        out = {"reply": f"⚠️ Something went wrong: `{type(e).__name__}: {e}`", "spoken": "Something went wrong. Please check the message on screen.", "tools": []}
    idx = len(S.messages)
    audio = speak(out.get("spoken") or out["reply"], idx)
    S.messages.append({"role": "assistant", "content": out["reply"], "audio": audio, "tools": out.get("tools", [])})


tab_chat, tab_proc, tab_tasks, tab_hist = st.tabs(["💬 Assistant (chat + voice)", "📄 Process Meeting", "✅ Tasks", "🧠 Meeting memory"])

# ---- chat + voice
with tab_chat:
    for i, m in enumerate(S.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("tools"):
                with st.expander(f"🛠️ Agent used {len(m['tools'])} tool call(s)"):
                    for t in m["tools"]:
                        st.markdown(f"**{t['tool']}**"); st.json({"input": t["input"], "output": t["output"]}, expanded=False)
            if m.get("audio"):
                st.audio(m["audio"], format="audio/mp3", autoplay=(S.play_idx == i))
    S.play_idx = None

    c1, c2 = st.columns([1, 3])
    mic = c1.audio_input("🎙️ Tap to talk", key=f"mic_{S.mic_n}")
    c2.caption("Say: “What's pending?”, “Mark task 2 done”, “Assign task 3 to Karthik”, “Email me the last meeting summary”, "
               "“Remind Rahul about his tasks”. Replies come back as text **and** voice.")
    if mic is not None:
        data = mic.getvalue()
        h = hashlib.md5(data).hexdigest()
        if h != S.last_audio:
            S.last_audio = h
            with st.spinner("Transcribing…"):
                text, err = transcribe_groq(llm.client, data, tts_lang)
                if err:                                   # fallback: Google web speech
                    text, err = transcribe(data, stt_lang)
            if err:
                st.error(err)
            else:
                handle_user(text, via_voice=True)
                S.mic_n += 1
                st.rerun()
    if prompt := st.chat_input("Type a message or paste a transcript…"):
        handle_user(prompt)
        st.rerun()
    if S.speak_text:
        components.html(browser_speech_html(S.speak_text, stt_lang), height=30)
        S.speak_text = None

# ---- process meeting
with tab_proc:
    st.subheader("Run the agent pipeline")
    st.caption("Transcript Analyzer → Decision & Action → Task Assignment → Summary (A2A messages, all powered by Groq)")
    title = st.text_input("Meeting title", value="Weekly Product Sync")
    up = st.file_uploader("Upload transcript / notes", type=["txt", "md", "docx", "pdf", "vtt", "srt"])
    if st.button("Load sample transcript"):
        S["raw_text"] = open("sample_transcript.txt", encoding="utf-8").read()
    if up is not None and S.get("loaded_file") != up.name:
        S["raw_text"] = orch.parser.parse(up.name, up.getvalue())
        S["loaded_file"] = up.name
    raw = st.text_area("…or paste text", key="raw_text", height=220)

    if st.button("▶ Run agents", type="primary", disabled=not raw.strip()):
        try:
            with st.spinner("Agents working…"):
                res = orch.process_meeting(title, raw)
            S.last_result = res
            idx = len(S.messages)
            audio = speak(res["summary"]["spoken"], idx)
            S.play_idx = None
            S.messages.append({"role": "assistant", "content": res["summary"]["markdown"], "audio": audio, "tools": []})
            for k, n in email_after_processing(res):
                (st.success if n.get("sent") else st.warning)(f"{'Email sent' if n.get('sent') else 'Email failed'}: {k}" + (f" – {n.get('error')}" if n.get("error") else ""))
            if audio:
                st.audio(audio, format="audio/mp3", autoplay=True)
            elif S.speak_text:
                components.html(browser_speech_html(S.speak_text, stt_lang), height=30); S.speak_text = None
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")

    res = S.last_result
    if res:
        a, b, c = st.columns(3)
        a.metric("Decisions", len(res["bundle"]["decisions"]))
        b.metric("Action items", len(res["assignment"]["tasks"]))
        c.metric("Need owner", len(res["assignment"]["needs_owner"]))
        st.markdown(res["summary"]["markdown"])
        if res["assignment"]["tasks"]:
            st.dataframe(pd.DataFrame(res["assignment"]["tasks"])[["id", "task", "owner", "due", "priority", "assigned_by"]], use_container_width=True, hide_index=True)
        e1, e2 = st.columns(2)
        if e1.button("✉️ Email summary to me"):
            try:
                st.success(f"Sent to {emailer.my_email}") if orch.email_summary(res).get("sent") else st.warning("Not sent")
            except Exception as e:
                st.error(str(e))
        if e2.button("✉️ Email owners their tasks"):
            for o in orch.notify_owners(res):
                (st.success if o.get("sent") else st.warning)(f"{o['owner']}: {'sent' if o.get('sent') else o.get('error')}")
        with st.expander("🔗 A2A trace (agent handoffs)"):
            for t in res["trace"]:
                st.markdown(f"`{t['ts']}` **{t['sender']} → {t['receiver']}** · `{t['kind']}`")
                st.json(t["payload"], expanded=False)

# ---- tasks
with tab_tasks:
    st.subheader("Pending actions memory")
    rows = mem.list_tasks()
    if rows:
        df = pd.DataFrame(rows)[["id", "task", "owner", "due", "priority", "status", "meeting_id"]]
        flt = st.multiselect("Status", ["open", "in_progress", "done"], default=["open", "in_progress"])
        st.dataframe(df[df["status"].isin(flt)], use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        tid = c1.number_input("Task #", min_value=1, step=1)
        new_status = c2.selectbox("Set status", ["open", "in_progress", "done"])
        if c3.button("Update status"):
            st.success("Updated") if tools.update_status(int(tid), new_status) else st.warning("Task not found")
            st.rerun()
    else:
        st.info("No tasks yet. Process a meeting first.")

# ---- history
with tab_hist:
    st.subheader("Previous meetings")
    q = st.text_input("Search past meetings")
    for m in (mem.search_meetings(q, 10) if q else mem.recent_meetings(10)):
        with st.expander(f"#{m['id']} · {m['title']} · {m['date']}"):
            st.markdown(m["summary"] or "_No summary_")
