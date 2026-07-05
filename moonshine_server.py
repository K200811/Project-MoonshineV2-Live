#!/usr/bin/env python3
"""
Project Moonshine V2 — Mock WebSocket Backend Server
=====================================================
Broadcasts simulated pipeline data to ws://localhost:8765
Listens for inbound commands (e.g., TRIGGER_DFT) and prints them.

JSON FILE MODE
--------------
To broadcast a static JSON file instead of generated mock data, either:

  1. Pass the filename as a CLI argument:
        python moonshine_server.py my_data.json

  2. Place a file named 'moonshine_data.json' in the same directory as
     this script — it will be picked up automatically.

The file is re-read from disk on every broadcast tick, so you can update
it while the server is running and the dashboard will receive the new data
on the next tick (every 3 seconds).

Requirements:
    pip install websockets

Usage:
    python moonshine_server.py                    # pure mock data
    python moonshine_server.py my_output.json     # broadcast JSON file
"""

import asyncio
import json
import random
import datetime
import sys
import threading
import os

try:
    import websockets
except ImportError:
    print("ERROR: 'websockets' package not found. Install it with:")
    print("  pip install websockets")
    sys.exit(1)

CONNECTED_CLIENTS = set()

# ── JSON file resolution ───────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON_NAME = "moonshine_data.json"

def resolve_json_file():
    """
    Return the path to a JSON file to broadcast, or None if mock data
    should be used instead.

    Priority:
      1. CLI argument  (python moonshine_server.py <filename>)
      2. 'moonshine_data.json' sitting next to this script
    """
    # 1. CLI argument
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        path = arg if os.path.isabs(arg) else os.path.join(SCRIPT_DIR, arg)
        if not os.path.isfile(path):
            print(f"ERROR: JSON file not found: {path}")
            sys.exit(1)
        return path

    # 2. Convention file next to the script
    default_path = os.path.join(SCRIPT_DIR, DEFAULT_JSON_NAME)
    if os.path.isfile(default_path):
        return default_path

    print("ERROR: No JSON file found. Provide one in either of these ways:")
    print(f"  1. Pass it as an argument:  python {os.path.basename(__file__)} <filename>.json")
    print(f"  2. Place a file named '{DEFAULT_JSON_NAME}' next to this script")
    sys.exit(1)

JSON_FILE_PATH = resolve_json_file()

# ── Payload source ─────────────────────────────────────────────────────────────
def load_json_payload():
    """
    Read JSON_FILE_PATH from disk and return the parsed object.
    Re-reads every tick so edits to the file are picked up live.
    """
    try:
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[!] JSON parse error in {JSON_FILE_PATH}: {e}")
        return {"error": "invalid JSON", "detail": str(e)}
    except OSError as e:
        print(f"[!] Could not read {JSON_FILE_PATH}: {e}")
        return {"error": "file read error", "detail": str(e)}


# ── Mock data (used only when no JSON file is present) ────────────────────────
FORMULAS = [
    "SrTiO3", "BaTiO3", "CaTiO3", "PbTiO3", "MgSiO3",
    "LiNbO3", "KNbO3", "NaNbO3", "BiFeO3", "LaAlO3",
    "SrVO3", "CaZrO3", "BaZrO3", "SrSnO3", "CaSnO3",
]

STAGES = [
    "Data Ingestion",
    "Structure Generation (PyXtal)",
    "ML Pre-Screening (CGCNN)",
    "Structure Relaxation (CHGNet)",
    "Energy Filtering",
    "Symmetry Analysis",
    "DFT Validation (VASP)",
    "Post-Processing & Archival",
]

STAGE_TIMINGS_TEMPLATE = [
    {"stage": "Data Ingestion", "seconds": 0},
    {"stage": "ML Pre-Screening", "seconds": 0},
    {"stage": "Structure Generation", "seconds": 0},
    {"stage": "ML Relaxation", "seconds": 0},
    {"stage": "Energy Filtering", "seconds": 0},
    {"stage": "DFT Validation", "seconds": 0},
]

WARNINGS = [
    "Warning: High VRAM allocation detected",
    "API latency spike on DFT cluster node-03",
    "Disk I/O bottleneck on /data/scratch",
    "Memory pressure above 85% threshold",
    "CHGNet inference queue depth > 50",
    "VASP job timeout on candidate batch 12",
    "Stale lock file detected in /tmp/moonshine",
    "GPU thermal throttling on device cuda:1",
]

FILES_TEMPLATE = [
    "structure_relaxed_{idx}.cif",
    "energy_profile_{idx}.json",
    "band_structure_{idx}.dat",
    "density_matrix_{idx}.npy",
    "report_{idx}.md",
    "symmetry_analysis_{idx}.json",
]

candidate_counter = 40
epoch_counter = 0
max_epochs = 100
train_loss = 0.8
test_loss = 0.9
candidates_processed = 1200
error_count = 2
start_time = datetime.datetime.now()


def build_payload():
    global candidate_counter, epoch_counter, train_loss, test_loss
    global candidates_processed, error_count

    candidate_counter += 1
    epoch_counter = min(epoch_counter + 1, max_epochs)
    candidates_processed += random.randint(1, 5)
    error_count += random.choice([0, 0, 0, 0, 1])

    # Decay losses with noise
    if epoch_counter < max_epochs:
        train_loss = max(0.005, train_loss * 0.97 + random.uniform(-0.005, 0.005))
        test_loss = max(0.01, test_loss * 0.965 + random.uniform(-0.008, 0.008))

    current_stage = random.choice(STAGES)
    idx = str(candidate_counter).zfill(4)

    # Generate 2-4 candidates per tick
    num_candidates = random.randint(2, 4)
    candidates = []
    for i in range(num_candidates):
        c_idx = str(candidate_counter + i).zfill(4)
        formula = random.choice(FORMULAS)
        status = random.choice(["processing", "processing", "processing", "queued", "completed", "failed_relaxation"])
        candidates.append({
            "formula": formula,
            "index": c_idx,
            "id": f"cand_{c_idx}",
            "status": status,
        })

    # Stage timing with variation
    stage_timing = []
    for st in STAGE_TIMINGS_TEMPLATE:
        stage_timing.append({
            "stage": st["stage"],
            "seconds": round(random.uniform(5, 300), 1),
        })

    # Elapsed time
    elapsed = datetime.datetime.now() - start_time
    hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Logs
    num_warnings = random.randint(0, 3)
    warnings = random.sample(WARNINGS, min(num_warnings, len(WARNINGS)))

    num_files = random.randint(0, 3)
    new_files = []
    for _ in range(num_files):
        tpl = random.choice(FILES_TEMPLATE)
        new_files.append(tpl.format(idx=idx))

    # Pass/fail candidates
    passed = [f"cand_{str(random.randint(1, candidate_counter)).zfill(4)}" for _ in range(random.randint(0, 3))]
    failed = [f"cand_{str(random.randint(1, candidate_counter)).zfill(4)}" for _ in range(random.randint(0, 2))]

    # CPU/Storage/VRAM with smooth variation
    cpu = round(random.uniform(20, 95), 1)
    storage = round(min(99, 60 + candidate_counter * 0.05 + random.uniform(-5, 5)), 1)
    vram = round(random.uniform(50, 98), 1)

    payload = {
        "packet_type": "UI_UPDATE",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "target_page": random.choice(["HOME", "SEARCH", "ANALYTICS", "LOGS"]),
        "payload": {
            "current_stage": current_stage,
            "candidates_in_system": candidates,
            "metrics": {
                "cpu_percent": cpu,
                "storage_percent": storage,
                "storage_total": "1TB",
                "vram_percent": vram,
                "vram_total": "68GB",
                "candidates_processed": candidates_processed,
                "error_count": error_count,
            },
            "stage_timing": stage_timing,
            "model_training": {
                "epoch": epoch_counter,
                "max_epochs": max_epochs,
                "train_loss": round(train_loss, 4),
                "test_loss": round(test_loss, 4),
                "elapsed_time": elapsed_str,
                "config": "Batch Size: 32, LR: 0.001, Optimizer: AdamW\nArchitecture: CGCNN-v2\nLayers: [64, 128, 256, 128]\nDropout: 0.15\nScheduler: CosineAnnealingWarmRestarts",
            },
            "logs": {
                "warnings": warnings,
                "new_files": new_files,
                "passed_all": passed,
                "failed_all": failed,
            },
        },
    }
    return payload


# ── WebSocket handler ──────────────────────────────────────────────────────────
async def handler(websocket):
    """Handle a single WebSocket connection."""
    CONNECTED_CLIENTS.add(websocket)
    remote = websocket.remote_address
    print(f"[+] Client connected: {remote}")

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                print(f"\n{'='*60}")
                print(f"[INBOUND COMMAND FROM BROWSER]")
                print(json.dumps(data, indent=2))
                print(f"{'='*60}\n")

                if data.get("action") == "TRIGGER_DFT":
                    cid = data.get("candidate_id", "unknown")
                    print(f">>> DFT TRIGGER received for candidate: {cid}")
                    print(f">>> Simulating DFT job submission...")

            except json.JSONDecodeError:
                print(f"[!] Non-JSON message received: {message}")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        CONNECTED_CLIENTS.discard(websocket)
        print(f"[-] Client disconnected: {remote}")


async def broadcast_loop():
    """Periodically broadcast data to all connected clients."""
    while True:
        if CONNECTED_CLIENTS:
            payload = load_json_payload()
            message = json.dumps(payload)
            disconnected = set()
            for client in CONNECTED_CLIENTS:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            CONNECTED_CLIENTS.difference_update(disconnected)
            print(f"[broadcast] Sent JSON file to {len(CONNECTED_CLIENTS)} client(s) | {os.path.basename(JSON_FILE_PATH)}")

        await asyncio.sleep(3)


# ── HTTP file server ───────────────────────────────────────────────────────────
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler

FILES_DIR = os.path.join(SCRIPT_DIR, "shared_files")
os.makedirs(FILES_DIR, exist_ok=True)


class FileServerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FILES_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, format, *args):
        pass  # silence default logging


def start_file_server():
    server = HTTPServer(("localhost", 8766), FileServerHandler)
    print(f"[file-server] Serving files from {FILES_DIR}")
    print(f"[file-server] Listening on http://localhost:8766/files/<name>")
    server.serve_forever()


async def main():
    print("=" * 60)
    print("  PROJECT MOONSHINE V2 — WebSocket Server")
    print("  Listening on ws://localhost:8765")
    print(f"  JSON FILE: {os.path.basename(JSON_FILE_PATH)}")
    print(f"  (edit the file while running — changes apply next tick)")
    print("=" * 60)
    print()

    # Start HTTP file server in a background thread
    file_thread = threading.Thread(target=start_file_server, daemon=True)
    file_thread.start()

    async with websockets.serve(handler, "localhost", 8765):
        await broadcast_loop()


if __name__ == "__main__":
    asyncio.run(main())