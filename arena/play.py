"""Let Laya (a text classifier) play ViZDoom's defend_the_center.

Each tic: game state -> one sentence -> Laya picks 1 of 3 actions.
Usage: .venv/bin/python play.py [--policy laya|bot|random] [--episodes 3] [--watch]
"""
import argparse
import base64
import io
import json
import os
import queue
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import laya
import vizdoom as vzd

CFG = os.path.join(os.path.dirname(vzd.__file__), "scenarios", "defend_the_center.cfg")
ACTIONS = {"turn_left": [1, 0, 0], "turn_right": [0, 1, 0], "shoot": [0, 0, 1]}
NOT_ENEMY = {"DoomPlayer", "Clip", "ClipBox", "Medikit", "Stimpack", "BulletPuff", "Blood", "DeadZombieMan"}
QUESTIONS = {"action": {
    "type": "choice",
    "instructions": "Where is the nearest enemy relative to the crosshair?",
    "criteria": {
        "shoot": "enemy is centred",
        "turn_left": "enemy is on the left side, or there is no enemy",
        "turn_right": "enemy is on the right side",
    },
}}


def enemies(state, width):
    """Screen-space enemies as (offset from centre in px, apparent size), nearest (biggest) first."""
    out = [(l.x + l.width / 2 - width / 2, l.width) for l in state.labels if l.object_name not in NOT_ENEMY]
    return sorted(out, key=lambda e: -e[1])


def describe(state, width):
    # Short symbolic text: Laya reads words ("FAR LEFT") far better than pixel numbers.
    ens = enemies(state, width)
    if not ens:
        return "No enemy visible."
    off = ens[0][0]
    if abs(off) < 12:
        return "Nearest enemy: CENTRE."
    return f"Nearest enemy: {'FAR ' if abs(off) > 50 else ''}{'LEFT' if off < 0 else 'RIGHT'}."


def bot(state, width):
    """Scripted aimbot: the teacher/baseline."""
    ens = enemies(state, width)
    if not ens:
        return "turn_left"
    off = ens[0][0]
    return "shoot" if abs(off) < 12 else ("turn_left" if off < 0 else "turn_right")


LIVE = {}  # latest snapshot, read by the dashboard server thread
REQ = queue.Queue()  # episode numbers picked in the dashboard
STOP = threading.Event()  # set by the dashboard's Stop button


def serve(port):
    html = open(os.path.join(os.path.dirname(__file__), "dash.html"), "rb").read()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body, ctype = (json.dumps(LIVE).encode(), "application/json") if self.path == "/state" else (html, "text/html")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # /play?ep=N or /stop
            if self.path == "/stop":
                STOP.set()
                ok = True
            else:
                ep = int(self.path.rpartition("=")[2])
                ok = LIVE.get("status") != "playing" and 1 <= ep <= LIVE["n"]
                if ok:
                    REQ.put(ep)
            self.send_response(204 if ok else 409)
            self.end_headers()

        def log_message(self, *_):
            pass

    threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever, daemon=True).start()
    print(f"dashboard: http://127.0.0.1:{port}")


def jpeg_b64(state):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(state.screen_buffer.transpose(1, 2, 0)).save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def make_laya():

    agent = laya.load("convaiinnovations/laya")

    def policy(state, width):
        text = describe(state, width)
        r = agent.predict(text, QUESTIONS)["answers"]["action"]
        LIVE.update(text=text, probs=r["probabilities"], conf=r["answer_confidence"])
        return r["choice"]
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="laya", choices=["laya", "bot", "random"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--watch", action="store_true", help="show the game window")
    ap.add_argument("--dash", type=int, nargs="?", const=8000, help="serve live dashboard on this port (default 8000)")
    a = ap.parse_args()
    if a.dash:
        serve(a.dash)

    g = vzd.DoomGame()
    g.load_config(CFG)
    g.set_labels_buffer_enabled(True)
    g.set_window_visible(a.watch)
    g.set_mode(vzd.Mode.ASYNC_PLAYER if a.watch else vzd.Mode.PLAYER)
    g.init()
    width = g.get_screen_width()
    policy = {"laya": make_laya, "bot": lambda: bot, "random": lambda: lambda s, w: random.choice(list(ACTIONS))}[a.policy]()

    def run_episode(ep):
        """Play episode `ep` (1-based). The number is the seed, so an episode is the same map/spawns every time."""
        STOP.clear()
        g.set_seed(ep)
        g.new_episode()
        LIVE.pop("frame", None)  # don't show the previous episode's last frame
        LIVE.update(status="playing", run=LIVE.get("run", 0) + 1, episode=ep, kills=0, tic=0)
        tics = 0
        t0 = time.time()
        while not g.is_episode_finished() and not STOP.is_set():
            st = g.get_state()
            act = policy(st, width)
            if a.dash:
                LIVE.update(frame=jpeg_b64(st), action=act, tic=tics, ammo=int(st.game_variables[0]),
                            health=int(st.game_variables[1]), kills=int(g.get_game_variable(vzd.GameVariable.KILLCOUNT)))
            g.make_action(ACTIONS[act], 4)  # frame-skip 4: ~9 decisions/sec of game time
            tics += 1
        kills = int(g.get_game_variable(vzd.GameVariable.KILLCOUNT))
        LIVE.update(status="stopped" if STOP.is_set() else "done", kills=kills)
        print(f"ep {ep}: kills={kills} decisions={tics} {(time.time() - t0) / max(tics, 1) * 1000:.0f}ms/decision"
              + (" (stopped)" if STOP.is_set() else ""))

    if a.dash:
        LIVE.update(status="idle", n=a.episodes)
        while True:  # play only what the dashboard asks for
            run_episode(REQ.get())
    for ep in range(1, a.episodes + 1):
        run_episode(ep)
    g.close()


if __name__ == "__main__":
    main()
