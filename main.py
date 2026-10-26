"""
Arthur CLIENT - the sci-fi tkinter UI, microphone, speech and tools.
The AI itself runs on the Arthur server (server.py, next to Ollama).

Say "Arthur" followed by your request (e.g. "Arthur, what time is it?").
If you only say "Arthur", it answers "Yes?" and listens for your request.
Your words appear live as you speak. Tools run on THIS machine.

Setup:
    pip install vosk sounddevice pillow piper-tts
    python3 -m piper.download_voices en_GB-northern_english_male-medium
    Download a Vosk model from https://alphacephei.com/vosk/models
    (e.g. vosk-model-small-en-us-0.15), unzip it, and set VOSK_MODEL_PATH below.

Run:
    python3 client.py --server http://SERVER_IP:8000 --token mysecret
    (or set SERVER_URL / SERVER_TOKEN below, or the ARTHUR_SERVER / ARTHUR_TOKEN
    environment variables)

UI keys:  F11 = toggle fullscreen,  Esc = leave fullscreen
"""

import base64
import io
import argparse
import importlib.util
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import tkinter as tk
import wave
import webbrowser
from datetime import datetime
from tkinter import font as tkfont

import sounddevice as sd
from PIL import Image
from vosk import KaldiRecognizer, Model, SetLogLevel

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
# Where the Arthur server (server.py) is running
SERVER_URL = os.environ.get("ARTHUR_SERVER", "http://localhost:8000")
SERVER_TOKEN = os.environ.get("ARTHUR_TOKEN", "")  # must match the server's --token
REQUEST_TIMEOUT = 120  # seconds to wait for the server to send *something*
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000
SPEAK_REPLIES = True  # set False to only print replies
PIPER_VOICE = "en_GB-northern_english_male-medium"  # Piper voice used for speech
PIPER_VOICE_DIR = None  # folder with the voice .onnx/.json files (None = next to client.py, then the current folder)
PIPER_USE_CUDA = False  # True = run Piper on an NVIDIA GPU (pip install onnxruntime-gpu)
PIPER_LENGTH_SCALE = 1.0  # speaking speed: lower = faster (e.g. 0.85), higher = slower
MIC_UNMUTE_DELAY = 0.4  # seconds to keep the mic muted after Arthur stops talking (room echo)

# Words the speech recognizer might produce when you say "Arthur"
WAKE_WORDS = ["arthur", "author", "arthurs"]

# UI
BLUE = "#1043D8"
BG = "#000000"
IDLE_TIMEOUT_MS = 20000  # go back to the single "Please speak now..." panel after this
# Fonts are tried in order; the first one installed is used.
# (Orbitron / Aldrich / Bank Gothic give the closest look to the design.)
TECH_FONTS = ["Orbitron", "Aldrich", "Bank Gothic", "Eurostile", "Agency FB",
              "Consolas", "DejaVu Sans Mono", "Courier New"]
BACKGROUND_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ircu.png")


# ----------------------------------------------------------------------------
# TOOLS
#
# Three easy ways to add a tool. The AI reads each tool's description to decide
# when to use it. Tools in (1) and (2) are re-scanned on every request, so you
# do NOT need to restart Arthur after adding or editing one.
#
# 1) Drop a file into the  tools/  folder  (the file name = the tool name)
#
#      tools/lock_screen.sh      any script or executable. Put these in the
#                                first lines of the file:
#                                    # description: Lock the computer screen
#                                    # args: no          (yes = AI may pass text)
#                                    # background: no    (yes = launch and don't wait)
#
#      tools/weather.py          a Python file that defines:
#                                    DESCRIPTION = "Get the weather. Argument: city"
#                                    TAKES_ARGS = True          (optional)
#                                    def run(args): return "Sunny, 22 degrees"
#
#    Files starting with "_" are ignored (see the _example files for templates).
#
# 2) Add a line to  tools.json  (no code, good for existing commands/programs):
#      "open_calc": {"description": "Open the calculator", "run": "gnome-calculator",
#                    "background": true}
#    Optional keys: "takes_args" (true/false), "timeout" (seconds), "background".
#    "run" can be a command, or a script path (relative paths are relative to
#    this file's folder). .py runs with Python, .sh runs with bash.
#
# 3) Quick Python function right here in this file with the @tool decorator:
#      @tool("Say hello.")
#      def hello(): return "Hello!"
#
# Whatever a tool prints/returns is given back to the AI to summarize.
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(BASE_DIR, "tools")
TOOLS_JSON = os.path.join(BASE_DIR, "tools.json")

BUILTIN_TOOLS = {}  # filled by the @tool decorator
TOOLS = {}          # builtin + tools.json + tools/ folder (rebuilt by refresh_tools)


def tool(description, takes_args=False, name=None):
    """Decorator: register a Python function in this file as a tool."""
    def register(fn):
        BUILTIN_TOOLS[name or fn.__name__] = {
            "description": description, "run": fn, "takes_args": takes_args,
        }
        return fn
    return register


@tool("Get the current time and date.")
def get_time():
    return datetime.now().strftime("It is %I:%M %p on %A, %B %d.")


@tool("Open a website in the browser. Argument: the URL or site name. For YouTube searches use open_youtube instead.", takes_args=True)
def open_website(url):
    if not url.startswith("http"):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opened {url}"

@tool("Open a steam aplication")
def open_steam():
    steam_path = r"C:\Program Files (x86)\Steam\Steam.exe"
    subprocess.Popen(steam_path)
    return "Steam opened"

@tool("Search YouTube and open the results page in the browser. Use this whenever the "
      "user asks to open, search or look up something on YouTube. Argument: only the "
      "search words (for example 'minecraft videos'), without 'search for' or 'on YouTube'. "
      "Leave the argument empty to just open YouTube.", takes_args=True)
def open_youtube(query=""):
    query = query.strip()
    if not query:
        webbrowser.open("https://www.youtube.com")
        return "Opened YouTube."
    # quote_plus turns spaces into + : https://www.youtube.com/results?search_query=minecrafty+vid
    url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query)
    webbrowser.open(url)
    return f"Opened YouTube search results for {query}."


# --- automatic discovery ----------------------------------------------------
_warned = set()
_module_cache = {}  # path -> (mtime, module)


def _warn_once(key, message):
    if key not in _warned:
        _warned.add(key)
        print(f"[tools] {message}")


def _truthy(value):
    return str(value).strip().lower() in ("yes", "y", "true", "1", "on")


def _read_header(path):
    """Read '# description:', '# args:', '# background:' lines from a script."""
    info = {"description": None, "takes_args": False, "background": False}
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for _, line in zip(range(20), f):
                m = re.match(r"\s*(?:#|//|REM|::)\s*(description|args|background)\s*:\s*(.*)",
                             line, re.I)
                if not m:
                    continue
                key, value = m.group(1).lower(), m.group(2).strip()
                if key == "description":
                    info["description"] = value
                elif key == "args":
                    info["takes_args"] = _truthy(value)
                else:
                    info["background"] = _truthy(value)
    except OSError:
        pass
    return info


def _load_py_tool(path):
    mtime = os.path.getmtime(path)
    cached = _module_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    stem = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(f"arthur_tool_{stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _module_cache[path] = (mtime, module)
    return module


def discover_tools():
    found = {}

    # tools.json
    if os.path.isfile(TOOLS_JSON):
        try:
            with open(TOOLS_JSON, encoding="utf-8") as f:
                data = json.load(f)
            for name, spec in data.items():
                if name.startswith("_"):
                    continue  # lets you keep notes like "_help" in the file
                if not isinstance(spec, dict) or not spec.get("description") or not spec.get("run"):
                    _warn_once(f"json:{name}", f"tools.json entry '{name}' needs 'description' and 'run'")
                    continue
                found[name] = spec
        except Exception as e:
            _warn_once("json", f"could not read tools.json: {e}")

    # tools/ folder
    if os.path.isdir(TOOLS_DIR):
        for fname in sorted(os.listdir(TOOLS_DIR)):
            if fname.startswith(("_", ".")):
                continue
            path = os.path.join(TOOLS_DIR, fname)
            if not os.path.isfile(path):
                continue
            stem, ext = os.path.splitext(fname)
            try:
                if ext.lower() == ".py":
                    module = _load_py_tool(path)
                    found[stem] = {
                        "description": module.DESCRIPTION,
                        "run": module.run,
                        "takes_args": getattr(module, "TAKES_ARGS", False),
                    }
                else:
                    info = _read_header(path)
                    if not info["description"]:
                        _warn_once(path, f"{fname}: add a '# description: ...' line near the top to enable it")
                        continue
                    found[stem] = {
                        "description": info["description"],
                        "run": [path],
                        "takes_args": info["takes_args"],
                        "background": info["background"],
                    }
            except Exception as e:
                _warn_once(f"{path}:{os.path.getmtime(path)}", f"skipped {fname}: {e}")
    return found


def refresh_tools():
    """Rebuild TOOLS from the built-ins, tools.json and the tools/ folder."""
    global TOOLS
    TOOLS = {**BUILTIN_TOOLS, **discover_tools()}
    return TOOLS


# ----------------------------------------------------------------------------
# TOOL RUNNER
# ----------------------------------------------------------------------------
def run_tool(name, args=""):
    tool_def = TOOLS.get(name)
    if tool_def is None:
        return f"Unknown tool: {name}"

    target = tool_def["run"]
    takes_args = tool_def.get("takes_args", False)

    try:
        if callable(target):
            result = target(args) if takes_args else target()
            return str(result) if result is not None else "Done."

        if isinstance(target, (list, tuple)):
            cmd = list(target)
        else:
            cmd = shlex.split(target, posix=(os.name != "nt"))

        # relative paths are relative to this file's folder
        if not os.path.isabs(cmd[0]) and os.path.exists(os.path.join(BASE_DIR, cmd[0])):
            cmd[0] = os.path.join(BASE_DIR, cmd[0])

        lower = cmd[0].lower()
        if lower.endswith(".py"):
            cmd = [sys.executable] + cmd
        elif lower.endswith(".sh"):
            cmd = ["bash"] + cmd
        if takes_args and args:
            cmd.append(args)

        if tool_def.get("background"):
            kwargs = {"start_new_session": True} if os.name != "nt" else {}
            subprocess.Popen(cmd, cwd=BASE_DIR, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, **kwargs)
            return "Started."

        proc = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True,
                              timeout=tool_def.get("timeout", 60))
        output = (proc.stdout or "").strip() or (proc.stderr or "").strip()
        return output or "Done."
    except Exception as e:
        return f"Tool '{name}' failed: {e}"


# ----------------------------------------------------------------------------
# AI  (runs on the server - this part just talks to it)
# ----------------------------------------------------------------------------
class ServerError(Exception):
    """The Arthur server couldn't be reached or reported an error."""


def _headers():
    headers = {"Content-Type": "application/json"}
    if SERVER_TOKEN:
        headers["Authorization"] = f"Bearer {SERVER_TOKEN}"
    return headers


def stream_post(path, payload):
    """POST JSON to the server and yield each streamed event (a dict)."""
    req = urllib.request.Request(
        SERVER_URL.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers=_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            for line in resp:
                line = line.strip()
                if line:
                    yield json.loads(line)
    except urllib.error.HTTPError as e:
        raise ServerError(f"server returned HTTP {e.code} ({e.reason})") from e
    except (urllib.error.URLError, OSError) as e:
        raise ServerError(f"can't reach {SERVER_URL}: {e}") from e


def check_server():
    try:
        req = urllib.request.Request(SERVER_URL.rstrip("/") + "/health", headers=_headers())
        with urllib.request.urlopen(req, timeout=5) as resp:
            info = json.load(resp)
        print(f"[server] connected to {SERVER_URL} (model: {info.get('model')})")
    except Exception as e:
        print(f"[server] WARNING: can't reach {SERVER_URL}: {e}")


history = []


def _read_events(events, on_text):
    """Consume a stream of server events -> (final_text, tool_call_or_None)."""
    answer, tool_call = "", None
    for ev in events:
        kind = ev.get("type")
        if kind == "text":
            if on_text:
                on_text(ev.get("text", ""))
        elif kind == "tool_call":
            tool_call = ev
        elif kind == "done":
            answer = ev.get("text", "")
        elif kind == "error":
            raise ServerError(ev.get("message", "unknown server error"))
    return answer, tool_call


def ask_arthur(user_text, on_text=None):
    """Ask the server. If on_text is given, it's called with the reply-so-far
    each time new words arrive, so the UI can show it as it's being written."""
    refresh_tools()  # pick up any tools added since the last request
    tool_list = [{"name": n, "description": t["description"]} for n, t in TOOLS.items()]

    answer, call = _read_events(
        stream_post("/chat", {"text": user_text, "history": history[-6:], "tools": tool_list}),
        on_text,
    )

    if call:  # the AI wants a tool - it runs HERE, on the client
        tool = str(call.get("tool", ""))
        args = str(call.get("args", "") or "")
        print(f"[tool] {tool}({args!r})")
        result = run_tool(tool, args)
        print(f"[tool result] {result}")
        answer, _ = _read_events(
            stream_post("/summarize", {"text": user_text, "tool": tool, "result": result}),
            on_text,
        )

    answer = (answer or "").strip() or "Sorry, I didn't catch that."
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": answer})
    return answer


# ----------------------------------------------------------------------------
# SPEECH / VOICE LOOP (runs in a background thread)
# ----------------------------------------------------------------------------
def split_wake_word(text):
    """Return (heard_wake_word, command_after_it)."""
    words = text.lower().split()
    for i, w in enumerate(words):
        if w in WAKE_WORDS:
            return True, " ".join(words[i + 1:]).strip()
    return False, ""


def speak_with_piper(text):
    """Synthesize `text` with Piper into a temp WAV, then play it and block
    until playback has completely finished.

    (Piper's built-in player gives up after 5 seconds of draining, which cuts
    off longer replies and makes the mic un-mute too early.)
    """
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            [sys.executable, "-m", "piper", "-m", PIPER_VOICE, "-f", path, "--", text],
            check=True, timeout=120,
        )
        with wave.open(path, "rb") as wf:
            with sd.RawOutputStream(
                samplerate=wf.getframerate(), channels=wf.getnchannels(),
                dtype="int16",
            ) as out:
                while True:
                    frames = wf.readframes(4096)
                    if not frames:
                        break
                    out.write(frames)
            # leaving the `with` block waits for the audio to finish playing
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# Where one sentence ends: . ! ? followed by a space, or a line break
SENTENCE_END = re.compile(r"(?:[.!?…]+[\"')\]]*\s+|\n+)")


def find_voice_file():
    """Locate the Piper voice (.onnx) so it can be loaded into memory once."""
    if PIPER_VOICE.endswith(".onnx") and os.path.isfile(PIPER_VOICE):
        return PIPER_VOICE
    folders = [PIPER_VOICE_DIR, BASE_DIR, os.getcwd(),
               os.path.expanduser("~/.local/share/piper")]
    for folder in folders:
        if folder:
            path = os.path.join(folder, PIPER_VOICE + ".onnx")
            if os.path.isfile(path):
                return path
    return None


class Speaker:
    """Fast text-to-speech with Piper.

    Two things make this much quicker than running `python -m piper` per reply:
      1. The voice is loaded into memory ONCE at startup (not on every reply).
      2. Speech starts as soon as the first sentence is written - while the AI is
         still generating the rest - and the next sentence is synthesized while
         the current one is playing.

    Usage:  begin() -> feed(text_so_far) ... -> end(final_text)
    If the voice can't be loaded it falls back to the slow command-line method.
    """

    def __init__(self, on_start=None, on_stop=None):
        self.enabled = SPEAK_REPLIES
        self.on_start = on_start or (lambda: None)  # called just before the first sound
        self.on_stop = on_stop or (lambda: None)    # called after the last sound
        self.voice = None
        self.syn_config = None
        self._active = False

    def load(self):
        if not self.enabled:
            return
        path = find_voice_file()
        if not path:
            print(f"[piper] voice file '{PIPER_VOICE}.onnx' not found - using slower "
                  "command-line mode. Put the .onnx/.json files next to client.py "
                  "(or set PIPER_VOICE_DIR) for fast mode.")
            return
        try:
            from piper import PiperVoice
            print("Loading voice...")
            self.voice = PiperVoice.load(path, use_cuda=PIPER_USE_CUDA)
            if PIPER_LENGTH_SCALE != 1.0:
                from piper import SynthesisConfig
                self.syn_config = SynthesisConfig(length_scale=PIPER_LENGTH_SCALE)
            list(self._synth("Ready."))  # warm-up so the first real reply isn't slow
            print("Voice ready.")
        except Exception as e:
            print(f"[piper] fast mode unavailable ({e}); using slower command-line mode.")
            self.voice = None

    def _synth(self, text):
        if self.syn_config is not None:
            return self.voice.synthesize(text, syn_config=self.syn_config)
        return self.voice.synthesize(text)

    # ---- public API ----
    def begin(self):
        if not self.enabled:
            return
        self._active = True
        self._text, self._cut, self._started = "", 0, False
        if self.voice is None:
            return  # fallback mode: everything is spoken in end()
        self._sent_q = queue.Queue()
        self._audio_q = queue.Queue(maxsize=8)
        self._threads = [threading.Thread(target=self._synth_loop, daemon=True),
                         threading.Thread(target=self._play_loop, daemon=True)]
        for t in self._threads:
            t.start()

    def feed(self, text):
        """`text` is everything written so far (not just the new bit)."""
        if self._active and self.voice is not None:
            self._text = text
            self._queue_sentences(final=False)

    def end(self, final_text=None):
        """Speak whatever is left, wait until the audio has finished, un-mute."""
        if not self._active:
            return
        self._active = False

        if self.voice is None:  # fallback: the old slower method
            text = (final_text or "").strip()
            if text:
                self.on_start()
                try:
                    speak_with_piper(text)
                except Exception as e:
                    print(f"[piper error] {e}")
                finally:
                    self.on_stop()
            return

        # trust the final text only if it continues what was already queued
        if final_text is not None and final_text.startswith(self._text[:self._cut]):
            self._text = final_text
        self._queue_sentences(final=True)
        self._sent_q.put(None)
        for t in self._threads:
            t.join()

    # ---- internals ----
    def _queue_sentences(self, final):
        rest = self._text[self._cut:]
        while True:
            m = SENTENCE_END.search(rest)
            if not m:
                break
            sentence = rest[:m.end()].strip()
            self._cut += m.end()
            rest = rest[m.end():]
            if sentence:
                self._sent_q.put(sentence)
        if final:
            self._cut = len(self._text)
            if rest.strip():
                self._sent_q.put(rest.strip())

    def _synth_loop(self):
        try:
            while True:
                sentence = self._sent_q.get()
                if sentence is None:
                    break
                for chunk in self._synth(sentence):
                    self._audio_q.put((chunk.sample_rate, chunk.sample_channels,
                                       chunk.audio_int16_bytes))
        except Exception as e:
            print(f"[piper error] {e}")
        finally:
            self._audio_q.put(None)

    def _play_loop(self):
        stream, failed = None, False
        try:
            while True:
                item = self._audio_q.get()
                if item is None:
                    break
                if failed:
                    continue  # keep draining so the synth thread never blocks
                try:
                    rate, channels, data = item
                    if stream is None:
                        self._started = True
                        self.on_start()  # mute the mic right before the first sound
                        stream = sd.RawOutputStream(samplerate=rate, channels=channels,
                                                    dtype="int16")
                        stream.start()
                    stream.write(data)
                except Exception as e:
                    print(f"[audio error] {e}")
                    failed = True
        finally:
            if stream is not None:
                try:
                    stream.stop()  # waits until everything has actually been played
                    stream.close()
                except Exception:
                    pass
            if self._started:
                self.on_stop()


def voice_loop(ui_q, stop):
    """Listens for the wake word, talks to Arthur, and posts updates to the UI.

    ui_q messages:
        ("transcript", text)  - what the user said
        ("response", text)    - what Arthur says
        ("idle", None)        - nothing happening (not used here, UI times out itself)
    """
    muted = threading.Event()  # while set, microphone audio is thrown away

    def mic_off():
        muted.set()  # Arthur is about to talk

    def mic_on():
        time.sleep(MIC_UNMUTE_DELAY)  # let the room echo die down
        while not audio_q.empty():
            audio_q.get_nowait()
        recognizer.Reset()
        muted.clear()  # mic is live again

    speaker = Speaker(on_start=mic_off, on_stop=mic_on)

    def speak(text):
        """Say a complete piece of text (e.g. "Yes?")."""
        print(f"Arthur: {text}")
        ui_q.put(("response", text))
        speaker.begin()
        speaker.end(text)

    def answer_command(command):
        """Ask the server and show + speak the reply while it is being written."""
        ui_q.put(("response", "Thinking..."))
        speaker.begin()

        def on_text(text):
            ui_q.put(("response", text))
            speaker.feed(text)

        try:
            answer = ask_arthur(command, on_text=on_text)
        except Exception:
            speaker.end()  # finish/un-mute cleanly, then let the caller report the error
            raise
        print(f"Arthur: {answer}")
        ui_q.put(("response", answer))
        speaker.end(answer)

    SetLogLevel(-1)
    rec_end = 0.45 if SPEAK_REPLIES else 0.95
    ui_q.put(("loading", (0.10, rec_end, "Loading speech recognition...")))
    recognizer = KaldiRecognizer(Model(VOSK_MODEL_PATH), SAMPLE_RATE)
    audio_q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if not muted.is_set():
            audio_q.put(bytes(indata))

    if SPEAK_REPLIES:
        ui_q.put(("loading", (0.45, 0.95, "Loading voice...")))
    speaker.load()  # load the voice into memory once
    ui_q.put(("ready", None))

    awaiting_command = False
    last_partial = ""
    print("Listening... say 'Arthur' to wake me.")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, blocksize=4000, dtype="int16",
        channels=1, callback=callback,
    ):
        while not stop.is_set():
            try:
                data = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if not recognizer.AcceptWaveform(data):
                # Still mid-sentence: show the words so far, live
                partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()
                if partial and partial != last_partial:
                    last_partial = partial
                    if awaiting_command or split_wake_word(partial)[0]:
                        ui_q.put(("transcript", partial))
                continue
            last_partial = ""

            text = json.loads(recognizer.Result()).get("text", "").strip()
            if not text:
                continue
            print(f"You: {text}")

            command = None
            if awaiting_command:
                command = text
                awaiting_command = False
                ui_q.put(("transcript", text))
            else:
                woke, rest = split_wake_word(text)
                if not woke:
                    continue
                ui_q.put(("transcript", text))
                if rest:
                    command = rest
                else:
                    ui_q.put(("transcript", "Please speak now..."))
                    speak("Yes?")
                    awaiting_command = True

            if command:
                try:
                    answer_command(command)
                except ServerError as e:
                    speak("Sorry, I can't reach my server.")
                    print(f"[server error] {e}")
                except Exception as e:
                    speak("Sorry, something went wrong.")
                    print(f"[error] {e}")

            # Drop audio captured while Arthur was speaking so he doesn't hear himself
            while not audio_q.empty():
                audio_q.get_nowait()
            recognizer.Reset()
            last_partial = ""


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
DESIGN_W, DESIGN_H = 1456, 819  # panel coordinates below are in this 16:9 design space


class ArthurUI:
    def __init__(self, root, ui_q):
        self.root = root
        self.ui_q = ui_q
        self.mode = "loading"  # "loading" = startup bar, "idle" = one panel, "active" = two panels
        self.progress = 0.0            # loading bar, 0..1
        self.stage_start, self.stage_end = 0.0, 0.1
        self.loading_label = "Starting..."
        self.transcript = ""
        self.response = ""
        self._idle_job = None
        self._fonts = {}
        self._fullscreen = False

        installed = set(tkfont.families())
        self.tech = next((f for f in TECH_FONTS if f in installed), "Courier")

        # Background image (logo, circuit lines and ARTHUR-AI text are all in it)
        try:
            self.bg_src = Image.open(BACKGROUND_IMAGE).convert("RGB")
        except Exception as e:
            print(f"[ui] could not load background '{BACKGROUND_IMAGE}': {e}")
            self.bg_src = None
        self.bg_photo = None
        self._bg_size = None

        root.title("ARTHUR-AI")
        root.configure(bg=BG)
        root.geometry("1280x720")
        root.minsize(640, 360)

        self.canvas = tk.Canvas(root, bg=BG, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.draw())

        root.bind("<F11>", self.toggle_fullscreen)
        root.bind("<Escape>", self.exit_fullscreen)

        self.poll()

    # ---- window helpers ----------------------------------------------------
    def toggle_fullscreen(self, _=None):
        self._fullscreen = not self._fullscreen
        self.root.attributes("-fullscreen", self._fullscreen)

    def exit_fullscreen(self, _=None):
        self._fullscreen = False
        self.root.attributes("-fullscreen", False)

    # ---- state updates -----------------------------------------------------
    def poll(self):
        changed = False
        try:
            while True:
                kind, value = self.ui_q.get_nowait()
                if kind == "transcript":
                    self.mode = "active"
                    self.transcript = value
                    self.response = ""
                    self._arm_idle_timer()
                    changed = True
                elif kind == "response":
                    self.mode = "active"
                    self.response = value
                    self._arm_idle_timer()
                    changed = True
                elif kind == "idle":
                    self.go_idle()
                elif kind == "loading":
                    start, end, label = value
                    self.mode = "loading"
                    self.stage_start, self.stage_end = start, end
                    self.progress = max(self.progress, start)
                    self.loading_label = label
                    changed = True
                elif kind == "ready":
                    self.progress = 1.0
                    self.loading_label = "Ready."
                    changed = True
                    self.root.after(600, self._finish_loading)
        except queue.Empty:
            pass

        if self.mode == "loading":
            # Loading steps don't report real progress, so the bar creeps toward the
            # end of the current step and never quite reaches it until the step is done.
            if self.loading_label != "Ready.":
                ceiling = self.stage_end - 0.03 * (self.stage_end - self.stage_start)
                self.progress += (ceiling - self.progress) * 0.03
            changed = True
        if changed:
            self.draw()
        self.root.after(40, self.poll)

    def _arm_idle_timer(self):
        if self._idle_job:
            self.root.after_cancel(self._idle_job)
        self._idle_job = self.root.after(IDLE_TIMEOUT_MS, self.go_idle)

    def _finish_loading(self):
        if self.mode == "loading":  # unless the user already started talking
            self.go_idle()

    def go_idle(self):
        self._idle_job = None
        self.mode = "idle"
        self.transcript = ""
        self.response = ""
        self.draw()

    # ---- drawing -----------------------------------------------------------
    def font(self, family, px, weight="normal"):
        px = max(6, int(px))
        key = (family, px, weight)
        if key not in self._fonts:
            self._fonts[key] = tkfont.Font(family=family, size=-px, weight=weight)
        return self._fonts[key]

    def draw(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 50 or h < 50:
            return
        k = min(w / DESIGN_W, h / DESIGN_H)
        ox = (w - DESIGN_W * k) / 2
        oy = (h - DESIGN_H * k) / 2

        def P(x, y):
            return ox + x * k, oy + y * k

        # --- background image, scaled to fit (re-scaled only when size changes) ---
        if self.bg_src is not None:
            size = (max(1, int(DESIGN_W * k)), max(1, int(DESIGN_H * k)))
            if size != self._bg_size:
                buf = io.BytesIO()
                self.bg_src.resize(size, Image.BICUBIC).save(buf, format="PNG")
                self.bg_photo = tk.PhotoImage(
                    data=base64.b64encode(buf.getvalue()))
                self._bg_size = size
            c.create_image(ox, oy, image=self.bg_photo, anchor="nw")

        # --- panels ---
        if self.mode == "loading":
            self._loading_panel(P, k)
        elif self.mode == "idle":
            self._panel(P, k, 252, 1204, "TRANSCRIPTION", "Please speak now...")
        else:
            self._panel(P, k, 188, 687, "TRANSCRIPTION",
                        self.transcript or "Please speak now...")
            self._panel(P, k, 769, 1268, "ARTHUR RESPONSE", self.response)

    def _loading_panel(self, P, k):
        c = self.canvas
        self._panel(P, k, 252, 1204, "LOADING", self.loading_label)
        # progress bar: outlined frame with a solid fill inside
        left, top = P(284, 350)
        right, bottom = P(1172, 395)
        c.create_rectangle(left, top, right, bottom, outline=BLUE, width=max(2, 4 * k))
        pad = 6 * k
        fill_right = left + pad + (right - left - 2 * pad) * max(0.0, min(1.0, self.progress))
        if fill_right > left + pad + 1:
            c.create_rectangle(left + pad, top + pad, fill_right, bottom - pad,
                               fill=BLUE, outline=BLUE)
        c.create_text((left + right) / 2, P(0, 440)[1], text=f"{int(self.progress * 100)}%",
                      fill=BLUE, font=self.font(self.tech, 26 * k, "bold"))

    def _panel(self, P, k, x0, x1, title, body):
        c = self.canvas
        y0, y1, header = 213, 606, 249
        lw = max(2, 5 * k)
        left, top = P(x0, y0)
        right, bottom = P(x1, y1)
        c.create_rectangle(left, top, right, bottom, outline=BLUE, width=lw)
        hy = P(0, header)[1]
        c.create_line(left, hy, right, hy, fill=BLUE, width=lw)
        c.create_text((left + right) / 2, P(0, 231)[1], text=title, fill=BLUE,
                      font=self.font(self.tech, 24 * k, "bold"))
        pad = 8
        bx, by = P(x0 + pad + 4, header + 8)
        c.create_text(bx, by, text=body, anchor="nw", fill=BLUE,
                      width=(x1 - x0 - 2 * pad - 8) * k,
                      font=self.font(self.tech, 24 * k, "bold"))


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    global SERVER_URL, SERVER_TOKEN
    parser = argparse.ArgumentParser(description="Arthur client (UI + microphone + tools)")
    parser.add_argument("--server", default=SERVER_URL, help="Arthur server URL (default: %(default)s)")
    parser.add_argument("--token", default=SERVER_TOKEN, help="server password, if it has one")
    cli = parser.parse_args()
    SERVER_URL, SERVER_TOKEN = cli.server, cli.token

    if not os.path.isdir(VOSK_MODEL_PATH):
        sys.exit(
            f"Vosk model not found at '{VOSK_MODEL_PATH}'. Download one from "
            "https://alphacephei.com/vosk/models and update VOSK_MODEL_PATH."
        )

    refresh_tools()
    print("Tools loaded: " + ", ".join(TOOLS))

    ui_q = queue.Queue()
    stop = threading.Event()

    root = tk.Tk()
    ArthurUI(root, ui_q)

    def worker():
        ui_q.put(("loading", (0.0, 0.10, "Connecting to server...")))
        check_server()
        try:
            voice_loop(ui_q, stop)
        except Exception as e:
            print(f"[fatal] voice loop crashed: {e}")
            ui_q.put(("response", f"Error: {e}"))

    threading.Thread(target=worker, daemon=True).start()

    def on_close():
        stop.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        stop.set()
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
