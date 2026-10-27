"""Groq API wrapper (API key REQUIRED): chat, JSON mode, and a real tool-calling agent loop."""
import json
import os
import re


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, api_key=None, model=None, client=None):
        self.key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.client = client
        if self.client is None:
            if not self.key:
                raise LLMError("GROQ_API_KEY is required.")
            from groq import Groq
            self.client = Groq(api_key=self.key)

    def _chat(self, **kw):
        return self.client.chat.completions.create(model=self.model, **kw)

    def complete(self, system, user=None, messages=None, max_tokens=2000, json_mode=False):
        msgs = [{"role": "system", "content": system}] + (messages or [{"role": "user", "content": user}])
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        r = self._chat(messages=msgs, max_tokens=max_tokens, temperature=0.2, **extra)
        return (r.choices[0].message.content or "").strip()

    def complete_json(self, system, user, max_tokens=2500):
        raw = self.complete(system + "\nRespond with a single valid JSON object only.", user, max_tokens=max_tokens, json_mode=True)
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        s, e = raw.find("{"), raw.rfind("}")
        if s == -1 or e == -1:
            raise LLMError(f"Model did not return JSON: {raw[:200]}")
        return json.loads(raw[s:e + 1])

    def run_agent(self, system, messages, tools, handlers, max_steps=8, max_tokens=1500):
        """The model decides which tools to call; we execute them and loop until it answers.
        `tools` are given as {name, description, input_schema}; converted to Groq/OpenAI function format.
        Returns (final_text, tool_trace)."""
        oa_tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in tools]
        msgs = [{"role": "system", "content": system}] + list(messages)
        trace = []
        for _ in range(max_steps):
            try:
                r = self._chat(messages=msgs, tools=oa_tools, tool_choice="auto", max_tokens=max_tokens, temperature=0.2)
            except Exception as ex:
                if "tool_use_failed" in str(ex):      # model produced a malformed tool call: let it retry
                    msgs.append({"role": "user", "content": "Your last tool call was malformed. Retry with valid arguments."})
                    continue
                raise
            m = r.choices[0].message
            if not getattr(m, "tool_calls", None):
                return (m.content or "").strip(), trace
            msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}} for tc in m.tool_calls]})
            for tc in m.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    out = handlers[tc.function.name](**args)
                except Exception as ex:
                    args, out = {}, {"error": f"{type(ex).__name__}: {ex}"}
                trace.append({"tool": tc.function.name, "input": args, "output": out})
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, default=str)[:20000]})
        return "I stopped after too many steps. Please rephrase or narrow the request.", trace
