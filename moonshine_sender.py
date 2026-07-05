#!/usr/bin/env python3
"""
Moonshine Monitor — WebSocket Sender
====================================
Drop this file next to your algorithm script and use it to push live updates
to the dashboard at ws://localhost:8765.

Install once:
    pip install websockets

Usage in your pipeline:
    from moonshine_sender import MoonshineSender

    sender = MoonshineSender()
    sender.connect()

    # ...inside your training / processing loop...
    sender.send({
        "current_stage": "ML Pre-Screening",
        "candidates_in_system": [
            {"formula": "SrTiO3", "id": "cand_0001", "index": "0001", "status": "processing"},
        ],
        "metrics": {
            "cpu_percent": 72.4,
            "storage_percent": 65.1,
            "storage_total": "1TB",
            "vram_percent": 88.0,
            "vram_total": "68GB",
            "candidates_processed": 1200,
            "error_count": 2,
        },
        "stage_timing": [
            {"stage": "Data Ingestion", "seconds": 12.4},
            {"stage": "ML Pre-Screening", "seconds": 45.2},
        ],
        "model_training": {
            "epoch": 42,
            "max_epochs": 100,
            "train_loss": 0.034,
            "test_loss": 0.051,
            "elapsed_time": "00:12:03",
            "config": "Batch Size: 32, LR: 0.001",
        },
        "logs": {
            "warnings": ["High VRAM allocation detected"],
            "new_files": ["structure_relaxed_0001.cif"],
            "passed_all": ["cand_0001"],
            "failed_all": ["cand_0002"],
        },
    })

    sender.close()

You only need to include the fields you have — the dashboard merges partial
updates. See the FIELD REFERENCE below for every accepted field.
"""

import asyncio
import json
import datetime
import threading

try:
    import websockets
except ImportError:
    raise ImportError(
        "The 'websockets' package is required. Install it with:\n"
        "  pip install websockets"
    )


# ─────────────────────────────────────────────────────────────────
# FIELD REFERENCE  (all fields are optional — send only what you have)
# ─────────────────────────────────────────────────────────────────
#
# payload = {
#     "current_stage": str,                    # e.g. "ML Pre-Screening"
#
#     "candidates_in_system": [                # list of candidates currently active
#         {
#             "formula": str,                 # e.g. "SrTiO3"
#             "id": str,                      # unique id, e.g. "cand_0001"
#             "index": str,                   # display index, e.g. "0001"
#             "status": str,                  # "processing" | "queued" | "completed" | "failed_relaxation"
#             "files": [                      # OPTIONAL — downloadable files for this candidate
#                 {"name": "structure.cif", "url": "http://localhost:8766/files/structure.cif"},
#                 {"name": "report.json", "url": "http://localhost:8766/files/report.json"},
#             ],
#         },
#         ...
#     ],
#
#     "metrics": {
#         "cpu_percent": float,                # 0–100
#         "storage_percent": float,            # 0–100
#         "storage_total": str,               # e.g. "1TB"
#         "vram_percent": float,               # 0–100
#         "vram_total": str,                  # e.g. "68GB"
#         "candidates_processed": int,
#         "error_count": int,
#     },
#
#     "stage_timing": [                        # per-stage duration in seconds
#         {"stage": str, "seconds": float},
#         ...
#     ],
#
#     "model_training": {
#         "epoch": int,
#         "max_epochs": int,
#         "train_loss": float,
#         "test_loss": float,
#         "elapsed_time": str,                 # "HH:MM:SS"
#         "config": str,                      # multi-line config text
#     },
#
#     "logs": {
#         "warnings": [str, ...],             # current active warnings
#         "new_files": [str, ...],            # filenames generated since last update (accumulates)
#         "passed_all": [str, ...],           # candidate IDs that passed (accumulates, deduped)
#         "failed_all": [str, ...],           # candidate IDs that failed (accumulates, deduped)
#     },
# }


class MoonshineSender:
    """
    Thread-safe WebSocket client that pushes JSON updates to the dashboard.

    Works from both async and sync code — send() detects the calling context
    and handles the event loop for you.
    """

    def __init__(self, uri="ws://localhost:8765"):
        self.uri = uri
        self._ws = None
        self._lock = threading.Lock()

    def connect(self):
        """Establish the WebSocket connection (sync, blocking until connected)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context — spawn a background thread
                self._connect_in_thread()
                return
        except RuntimeError:
            pass
        # Sync context — create a persistent loop in a background thread
        self._connect_in_thread()

    def _connect_in_thread(self):
        """Run the connection in a dedicated background thread."""
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()

        def _runner():
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._async_connect())

        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    async def _async_connect(self):
        try:
            self._ws = await websockets.connect(self.uri)
            print(f"[MoonshineSender] Connected to {self.uri}")
        except Exception as e:
            print(f"[MoonshineSender] Connection failed: {e}")
        finally:
            self._ready.set()

    def send(self, payload: dict):
        """
        Send a payload dict to the dashboard. Wraps it in the envelope
        {"packet_type": "UI_UPDATE", "timestamp": ..., "payload": ...}.

        Safe to call from sync code running alongside an async pipeline,
        or directly from within an async context.
        """
        message = {
            "packet_type": "UI_UPDATE",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "payload": payload,
        }

        # Try to use an existing running loop (async pipeline context)
        try:
            loop = asyncio.get_running_loop()
            asyncio.ensure_future(self._async_send(message), loop=loop)
            return
        except RuntimeError:
            pass

        # No running loop — dispatch to our background loop
        with self._lock:
            if self._ws is None:
                print("[MoonshineSender] Not connected — call connect() first")
                return
            asyncio.run_coroutine_threadsafe(
                self._async_send(message), self._loop
            )

    async def _async_send(self, message: dict):
        if self._ws is None:
            return
        try:
            await self._ws.send(json.dumps(message))
        except Exception as e:
            print(f"[MoonshineSender] Send failed: {e}")
            # Attempt reconnect
            try:
                self._ws = await websockets.connect(self.uri)
            except Exception:
                pass

    def close(self):
        """Close the connection cleanly."""
        if self._ws:
            try:
                asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────
# QUICK TEST — run this file directly to verify the dashboard connects
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    sender = MoonshineSender()
    sender.connect()
    time.sleep(1)  # let the connection establish

    print("[test] Sending sample update...")
    sender.send({
        "current_stage": "Data Ingestion",
        "candidates_in_system": [
            {"formula": "SrTiO3", "id": "cand_0001", "index": "0001", "status": "processing"},
            {"formula": "BaTiO3", "id": "cand_0002", "index": "0002", "status": "queued"},
        ],
        "metrics": {
            "cpu_percent": 45.2,
            "storage_percent": 30.0,
            "storage_total": "1TB",
            "vram_percent": 60.0,
            "vram_total": "68GB",
            "candidates_processed": 2,
            "error_count": 0,
        },
        "stage_timing": [
            {"stage": "Data Ingestion", "seconds": 5.2},
        ],
        "model_training": {
            "epoch": 1,
            "max_epochs": 100,
            "train_loss": 0.8,
            "test_loss": 0.9,
            "elapsed_time": "00:00:05",
            "config": "Test run",
        },
        "logs": {
            "warnings": [],
            "new_files": [],
            "passed_all": [],
            "failed_all": [],
        },
    })

    time.sleep(2)
    print("[test] Done. Check your dashboard — it should show live data.")
    sender.close()
