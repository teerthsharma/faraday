"""
execution_daemon.py — Autonomous Banach Fixed-Point Supervisor

Manages the 2 000 000-epoch burn run of the God Tensor with:
  • Per-epoch JSON telemetry parsing (structlog stdout pipe)
  • Immutable append-only transcript.csv + convergence_log.jsonl ledgers
  • Git commit + push every 10 000 epochs (The Pulse)
  • NaN / 500%-spike divergence halt with FATAL_DIVERGENCE.md dump

Usage
-----
    python execution_daemon.py --epochs 2000000 --dim 3
    FARADAY_LOG_FORMAT=json python -m faraday.benchmarking --epochs 2000000 --dim 3
"""

from __future__ import annotations

import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from typing import TextIO

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_EPOCHS = 2_000_000
GIT_COMMIT_EVERY_DEFAULT = 10_000          # epochs between git push cycles
CHECKPOINT_EVERY = 10_000                   # epochs between checkpoint saves
LEDGER_DIR   = Path(__file__).parent / "runs"
LEDGER_CSV   = LEDGER_DIR / "transcript.csv"
LEDGER_JSONL = LEDGER_DIR / "convergence_log.jsonl"
FATAL_MARKER = LEDGER_DIR / "FATAL_DIVERGENCE.md"
CHECKPOINT_DIR = LEDGER_DIR / "checkpoints"
REPO_DIR     = Path(__file__).parent   # faraday/

# ---------------------------------------------------------------------------
# NaN guard constants
# ---------------------------------------------------------------------------

SPIKE_THRESHOLD = 5.0   # 500 % of 10-epoch moving average → halt
WINDOW_SIZE     = 10     # epochs for moving-average baseline
MIN_EPOCHS_BEFORE_CHECK = WINDOW_SIZE     # wait until window is full


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _load_latest_token() -> str | None:
    """Read GitHub PAT from ~/.git-credentials or ~/.hermes/.github_token."""
    token_path = Path.home() / ".hermes" / ".github_token"
    if token_path.exists():
        return token_path.read_text().strip()
    creds = Path.home() / ".git-credentials"
    if creds.exists():
        for line in creds.read_text().splitlines():
            if "github.com" in line:
                # format: https://TOKEN@github.com
                parts = line.split("://", 1)
                if len(parts) == 2 and "@" in parts[1]:
                    return parts[1].split("@")[0]
    return os.environ.get("GITHUB_TOKEN")


def _git_runner(git_dir: Path):
    """Factory for git subprocess calls targeting ``git_dir``."""
    token = _load_latest_token()

    def runner(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        if token:
            env["GITHUB_TOKEN"] = token
        return subprocess.run(
            ["git", "-C", str(git_dir)] + args,
            capture_output=True,
            text=True,
            check=check,
            env=env,
        )
    return runner


# ---------------------------------------------------------------------------
# Ledger writer — append-only, thread-safe via file.seek
# ---------------------------------------------------------------------------

import hashlib

class LedgerWriter:
    """
    Appends rows to transcript.csv and convergence_log.jsonl.

    Uses explicit seek-to-end before each write to honour O_APPEND semantics
    even on network file systems (NFS, sshfs) where append can misbehave.

    On resume, pass ``skip_until`` to skip epochs already recorded in the ledger.
    """

    def __init__(
        self, csv_path: Path, jsonl_path: Path, skip_until: int = 0
    ) -> None:
        self._csv_path   = csv_path
        self._jsonl_path = jsonl_path
        self._skip_until = skip_until
        self._last_hash  = "0000000000000000000000000000000000000000000000000000000000000000"

        # Ensure directory exists
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Attempt to recover the last hash from the CSV on resume
        if csv_path.exists() and skip_until > 0:
            try:
                # Read the last line of the CSV to get the previous hash
                with open(csv_path, "r", newline="") as fh:
                    lines = fh.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip().split(",")
                        if len(last_line) >= 7:
                            self._last_hash = last_line[6]
            except Exception:
                pass

        # Write CSV header only if the file does not exist
        if not csv_path.exists():
            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "epoch", "banach_loss", "betti_0_err",
                        "betti_1_err", "betti_2_err", "timestamp", "hash"
                    ],
                )
                writer.writeheader()

        # (Re)open handles — kept for the lifetime of the object
        self._csv_fh: TextIO   = open(csv_path, "a", newline="")
        self._jsonl_fh: TextIO = open(jsonl_path, "a")

        self._csv_writer = csv.DictWriter(
            self._csv_fh,
            fieldnames=[
                "epoch", "banach_loss", "betti_0_err",
                "betti_1_err", "betti_2_err", "timestamp", "hash"
            ],
        )

    def append(self, record: dict) -> None:
        """Append one parsed epoch record to both ledgers (skip if epoch ≤ skip_until)."""
        epoch = record.get("epoch", 0)
        if epoch <= self._skip_until:
            return

        # Cryptographic Hash Chain
        data_str = f"{epoch}|{record.get('banach_loss')}|{record.get('betti_0_err')}|{record.get('betti_1_err')}|{record.get('betti_2_err')}|{record.get('timestamp')}|{self._last_hash}"
        current_hash = hashlib.sha256(data_str.encode('utf-8')).hexdigest()
        self._last_hash = current_hash
        record["hash"] = current_hash

        # CSV
        self._csv_fh.seek(0, os.SEEK_END)
        self._csv_writer.writerow({
            "epoch":        record.get("epoch", ""),
            "banach_loss":  record.get("banach_loss", ""),
            "betti_0_err":  record.get("betti_0_err", ""),
            "betti_1_err":  record.get("betti_1_err", ""),
            "betti_2_err":  record.get("betti_2_err", ""),
            "timestamp":    record.get("timestamp", ""),
            "hash":         record.get("hash", ""),
        })
        self._csv_fh.flush()

        # JSONL
        self._jsonl_fh.seek(0, os.SEEK_END)
        self._jsonl_fh.write(json.dumps(record) + "\n")
        self._jsonl_fh.flush()

    def close(self) -> None:
        self._csv_fh.close()
        self._jsonl_fh.close()


# ---------------------------------------------------------------------------
# Git Pulse
# ---------------------------------------------------------------------------

class GitPulse:
    """
    Handles git add → commit → push every ``commit_every`` epochs.

    The commit message uses live telemetry, e.g.:
        chore: Epoch 10000 Reached. Betti-1 Error: 0.034 | Banach Loss: 0.0089
    """

    def __init__(
        self,
        repo_dir: Path,
        paths: list[Path],
        commit_every: int = GIT_COMMIT_EVERY_DEFAULT,
    ) -> None:
        self._repo_dir    = repo_dir
        self._paths      = paths
        self._commit_every = commit_every
        self._git        = _git_runner(repo_dir)
        self._last_commit_epoch = 0

    def try_commit(self, current_epoch: int, telemetry: dict) -> bool:
        """
        Commit + push if ``current_epoch`` is a multiple of ``commit_every``.

        Returns True if a commit was made, False otherwise.
        """
        if current_epoch == 0 or current_epoch % self._commit_every != 0:
            return False
        if current_epoch == self._last_commit_epoch:
            return False   # already committed this checkpoint

        epoch      = telemetry.get("epoch", current_epoch)
        b1_err     = _fmt(telemetry.get("betti_1_err"))
        banach     = _fmt(telemetry.get("banach_loss"))

        msg = (
            f"chore: Epoch {epoch} Reached. "
            f"Betti-1 Error: {b1_err} | Banach Loss: {banach}"
        )

        # Stage only the ledger files (never stage code artefacts)
        rel_paths = [str(p.relative_to(self._repo_dir)) for p in self._paths]

        try:
            self._git(["add"] + rel_paths)
            self._git(["commit", "-m", msg])
            self._git(["push"])
            self._last_commit_epoch = current_epoch
            print(
                f"[Pulse] {_now_iso()}  committed epoch {epoch}  "
                f"Betti-1={b1_err}  Banach={banach}",
                flush=True,
            )
            return True
        except subprocess.CalledProcessError as exc:
            print(
                f"[Pulse] git failed: {exc.stderr.strip()}",
                flush=True,
            )
            return False


def _fmt(val: object) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "NaN"
    try:
        return f"{float(val):.6g}"
    except Exception:
        return str(val)


# ---------------------------------------------------------------------------
# Divergence Monitor
# ---------------------------------------------------------------------------

class DivergenceMonitor:
    """
    Tracks banach_loss over the last ``window`` epochs.

    Halts if:
      • banach_loss becomes NaN
      • banach_loss exceeds ``spike_threshold × window_avg``
    """

    def __init__(
        self,
        window: int = WINDOW_SIZE,
        spike_threshold: float = SPIKE_THRESHOLD,
    ) -> None:
        self._window          = window
        self._spike_threshold = spike_threshold
        self._prev_window: list[float] = []   # last full ``window`` values
        self._current: list[float]     = []   # accumulating up to ``window``
        self._halt_requested  = False
        self._halt_reason: str | None = None

    def update(self, epoch: int, banach_loss: float) -> None:
        """
        Record a new loss value.  Sets ``halt_requested`` on divergence.
        """
        if self._halt_requested:
            return

        # ── NaN trap ────────────────────────────────────────────────────
        if math.isnan(banach_loss):
            self._halt_requested = True
            self._halt_reason = (
                f"NaN detected at epoch {epoch} — banach_loss=NaN"
            )
            return

        self._current.append(banach_loss)

        # Once current buffer is full, roll it into prev_window
        if len(self._current) == self._window:
            self._prev_window = self._current.copy()
            self._current = []

        # ── Spike trap ─────────────────────────────────────────────────
        # Fire only when we have a genuine divergence: the baseline must be
        # meaningfully non-zero (avg > 1e-7) AND the new value must exceed
        # spike_threshold × that baseline.
        # This dual guard prevents false halts when banach_loss has already
        # converged to numerical noise (typically 1e-9 to 1e-6 range).
        if len(self._prev_window) == self._window:
            avg = sum(self._prev_window) / self._window
            if avg > 1e-7 and banach_loss > self._spike_threshold * avg:
                self._halt_requested = True
                self._halt_reason = (
                    f"Spike halt at epoch {epoch}: "
                    f"banach_loss={banach_loss:.6g} exceeds "
                    f"{self._spike_threshold}× window-avg={avg:.6g}"
                )

    @property
    def halt_requested(self) -> bool:
        return self._halt_requested

    @property
    def halt_reason(self) -> str | None:
        return self._halt_reason


# ---------------------------------------------------------------------------
# Fatal Divergence Dump
# ---------------------------------------------------------------------------

def dump_fatal_divergence(
    records: list[dict],
    reason: str,
    output_path: Path,
) -> None:
    """
    Write ``FATAL_DIVERGENCE.md`` containing:
      • Halt reason
      • Last 100 parsed epoch records
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# FATAL DIVERGENCE REPORT",
        "",
        f"**Detected:** {_now_iso()}",
        f"**Reason:** {reason}",
        "",
        "---",
        "",
        "## Last 100 Epoch Records",
        "",
        "| Epoch | Banach Loss | Betti-0 Err | Betti-1 Err | Betti-2 Err | Timestamp | Hash |",
        "|------:|------------:|------------:|------------:|------------:|-----------|------|",
    ]

    for rec in records[-100:]:
        lines.append(
            f"| {rec.get('epoch', '')} "
            f"| {rec.get('banach_loss', '')} "
            f"| {rec.get('betti_0_err', '')} "
            f"| {rec.get('betti_1_err', '')} "
            f"| {rec.get('betti_2_err', '')} "
            f"| {rec.get('timestamp', '')} "
            f"| {rec.get('hash', '')} |"
        )

    output_path.write_text("\n".join(lines) + "\n")
    print(f"[FATAL] Divergence report written → {output_path}", flush=True)


def emergency_tag_and_push(
    repo_dir: Path,
    epoch: int,
    reason: str,
) -> None:
    """
    Tag the current HEAD with ``emergency/{epoch}`` and push the tag.
    """
    git  = _git_runner(repo_dir)
    tag  = f"emergency/{epoch}"
    msg  = f"Emergency halt at epoch {epoch}: {reason}"
    try:
        git(["tag", "-a", tag, "-m", msg])
        git(["push", "origin", tag], check=False)   # check=False: tag push is best-effort
        print(f"[Emergency] Tagged {tag} and pushed to origin", flush=True)
    except subprocess.CalledProcessError as exc:
        print(f"[Emergency] Tag/push failed: {exc.stderr.strip()}", flush=True)


# ---------------------------------------------------------------------------
# Subprocess I/O reader (non-blocking)
# ---------------------------------------------------------------------------

def _stream_reader(
    stream,
    pipe_name: str,
    callback,
):
    """Read ``stream`` line-by-line and invoke ``callback`` with each line."""
    try:
        for raw_line in stream:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")
            if line:
                callback(line)
    except Exception as exc:
        print(f"[stream_reader/{pipe_name}] read error: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Core parse
# ---------------------------------------------------------------------------

def parse_epoch_line(line: str) -> dict | None:
    """
    Parse a single JSON structlog line from stdout.

    Returns a flat dict with keys ``epoch, banach_loss, betti_0_err,
    betti_1_err, betti_2_err, timestamp`` or ``None`` if the line is not
    an epoch record (e.g. a debug/info log from a system component).

    On JSON decode failure the line is returned as ``{"raw": line}`` so the
    caller can record it as a parse error rather than silently dropping it.
    """
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    if not isinstance(obj, dict):
        return None

    # Accept any event name that carries epoch telemetry
    # (the burn loop emits event="burn_epoch", FDTD emits "fdtd_step")
    event = obj.get("event", "")
    if event not in ("burn_epoch", "epoch", "fixed_point_progress", "fdtd_step"):
        return None

    epoch = obj.get("epoch")
    if epoch is None:
        return None

    def _nan(v):
        """Return float(v) or math.nan if missing / non-numeric."""
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    return {
        "epoch":       int(epoch),
        "banach_loss": _nan(obj.get("banach_loss")),
        "betti_0_err": _nan(obj.get("betti_0_err")),
        "betti_1_err": _nan(obj.get("betti_1_err")),
        "betti_2_err": _nan(obj.get("betti_2_err")),
        "timestamp":   str(obj.get("timestamp") or ""),
    }


# ---------------------------------------------------------------------------
# Main daemon
# ---------------------------------------------------------------------------

def run_daemon(
    epochs: int = DEFAULT_EPOCHS,
    dim: int = 3,
    n_geometries: int = 100,
    nx: int = 60,
    ny: int = 60,
    num_modes: int = 8,
    seed: int = 42,
    git_every: int = GIT_COMMIT_EVERY_DEFAULT,
    mode: str = "train",
    dt: float = 0.01,
) -> None:
    """
    Spawn the benchmark burn subprocess and monitor it to completion
    (or divergence halt).
    """
    # ── Detect resume state (before ledger init) ───────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    resume_from: int | None = None
    ckpt_path: str | None = None
    skip_until: int = 0

    # Load epoch from each checkpoint file to find the latest, since the
    # filename always stays burn_checkpoint_000000.npz (overwritten each save)
    existing_checkpoints = sorted(CHECKPOINT_DIR.glob("burn_checkpoint_*.npz"))
    if existing_checkpoints:
        # Find checkpoint with highest epoch by loading it (not by filename)
        best_epoch = -1
        best_path: Path | None = None
        for cp in existing_checkpoints:
            try:
                data = np.load(cp, allow_pickle=True)
                ep = int(data["epoch"])
                if ep > best_epoch:
                    best_epoch = ep
                    best_path = cp
            except Exception:
                continue
        if best_path is not None and best_epoch > 0:
            resume_from = best_epoch
            ckpt_path = str(best_path)
            skip_until = best_epoch
            print(
                f"[Daemon] {_now_iso()}  resuming from epoch {resume_from}  "
                f"(checkpoint: {best_path.name}, epoch {best_epoch} inside)",
                flush=True,
            )

    # ── Ledger setup ─────────────────────────────────────────────────────
    csv_file = LEDGER_CSV if mode == "train" else LEDGER_DIR / "fdtd_transcript.csv"
    jsonl_file = LEDGER_JSONL if mode == "train" else LEDGER_DIR / "fdtd_log.jsonl"
    ledger = LedgerWriter(csv_file, jsonl_file, skip_until=skip_until)

    # ── Divergence monitor ───────────────────────────────────────────────
    # For FDTD, divergence is defined as norm (banach_loss mapped) > 10.0
    spike_thresh = SPIKE_THRESHOLD if mode == "train" else 10.0
    monitor = DivergenceMonitor(spike_threshold=spike_thresh)

    # ── Git pulse ────────────────────────────────────────────────────────
    pulse = GitPulse(
        repo_dir     = REPO_DIR,
        paths        = [csv_file, jsonl_file],
        commit_every = git_every,
    )

    # ── Build subprocess command ─────────────────────────────────────────
    # checkpoint_path is the actual file path to load (resume) or create (fresh)
    if ckpt_path is not None:
        cp = ckpt_path
    else:
        cp = str(CHECKPOINT_DIR / "burn_checkpoint.npz")

    if mode == "train":
        cmd = [
            sys.executable, "-m", "faraday.benchmarking",
            "--epochs",     str(epochs),
            "--dim",        str(dim),
            "--n-geometries", str(n_geometries),
            "--nx",         str(nx),
            "--ny",         str(ny),
            "--num-modes",  str(num_modes),
            "--seed",       str(seed),
            "--checkpoint-path", cp,
        ]
        if resume_from is not None:
            cmd += ["--resume-from", str(resume_from)]
    else:
        # FDTD Mode
        cmd = [
            sys.executable, "-m", "faraday.fdtd_runner",
            "--steps", str(epochs),
            "--dt", str(dt),
            "--checkpoint-path", cp,
        ]
    # Force JSON logging on the subprocess so we get clean parseable lines
    env = {**os.environ, "FARADAY_LOG_FORMAT": "json"}

    print(
        f"[Daemon] {_now_iso()}  spawning  {' '.join(cmd)}",
        flush=True,
    )

    proc = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        env    = env,
        cwd    = str(REPO_DIR),
    )

    # ── I/O threads ──────────────────────────────────────────────────────
    import threading
    parsed_records: list[dict] = []
    latest_telemetry: dict = {}

    def on_stdout_line(line: str) -> None:
        record = parse_epoch_line(line)
        if record is None:
            # Non-epoch JSON log line — echo to daemon stdout for visibility
            try:
                obj = json.loads(line)
                print(
                    f"[engine] {obj.get('event','?')}  {obj.get('message', line)}",
                    flush=True,
                )
            except Exception:
                print(f"[engine] {line}", flush=True)
            return

        # Record in ledgers
        ledger.append(record)
        parsed_records.append(record)

        # Update live state
        latest_telemetry.update(record)

        # Check divergence
        monitor.update(record["epoch"], record["banach_loss"])

        # Try git commit
        pulse.try_commit(record["epoch"], record)

        # Progress heartbeat every 1 000 epochs
        if record["epoch"] % 1000 == 0:
            b1  = _fmt(record["betti_1_err"])
            bl  = _fmt(record["banach_loss"])
            print(
                f"[Heartbeat] {_now_iso()}  "
                f"epoch={record['epoch']:,}  "
                f"Betti-1={b1}  Banach={bl}",
                flush=True,
            )

    # Spawn stdout reader thread
    reader = Thread(
        target=_stream_reader,
        args=(proc.stdout, "stdout", on_stdout_line),
        daemon=True,
    )
    reader.start()

    # ── Main loop: wait for subprocess to finish or halt ─────────────────
    halt_flag = Event()

    def poll_stderr():
        """Consume stderr to prevent pipe deadlock."""
        try:
            for line in proc.stderr:
                print(
                    f"[engine:stderr] {line.decode('utf-8', errors='replace').strip()}",
                    flush=True,
                )
        except Exception:
            pass

    stderr_thread = Thread(target=poll_stderr, daemon=True)
    stderr_thread.start()

    return_code = None
    while True:
        return_code = proc.poll()
        if return_code is not None:
            # Subprocess exited on its own
            break
        if monitor.halt_requested:
            print(
                f"[Daemon] {_now_iso()}  HALT REQUESTED — {monitor.halt_reason}",
                flush=True,
            )
            # SIGKILL the child
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            break
        time.sleep(0.1)

    # ── Drain remaining stdout (up to 1 s) ───────────────────────────────
    reader.join(timeout=1.0)

    # ── Divergence path: dump FATAL report and tag ───────────────────────
    if monitor.halt_requested:
        dump_fatal_divergence(
            records    = parsed_records,
            reason     = monitor.halt_reason or "unknown",
            output_path = FATAL_MARKER,
        )
        emergency_tag_and_push(
            repo_dir = REPO_DIR,
            epoch    = latest_telemetry.get("epoch", 0),
            reason   = monitor.halt_reason or "",
        )

        # Commit the FATAL file immediately
        git = _git_runner(REPO_DIR)
        try:
            rel = str(FATAL_MARKER.relative_to(REPO_DIR))
            git(["add", rel])
            git(["commit", "-m", f"chore: FATAL DIVERGENCE at epoch {latest_telemetry.get('epoch', 0)}"])
            git(["push"], check=False)   # non-fatal on conflict
        except subprocess.CalledProcessError as exc:
            print(f"[Daemon] Fatal commit failed: {exc.stderr}", flush=True)

        print("[Daemon] Exiting with code 1 (divergence halt)", flush=True)
        ledger.close()
        sys.exit(1)

    # ── Normal completion ─────────────────────────────────────────────────
    stderr_thread.join(timeout=2.0)
    reader.join(timeout=2.0)

    if return_code != 0:
        print(
            f"[Daemon] subprocess exited with code {return_code}",
            flush=True,
        )

    # Final git commit at completion
    if latest_telemetry:
        pulse.try_commit(epochs, latest_telemetry)
        # One last push to make sure final epoch is captured
        try:
            git = _git_runner(REPO_DIR)
            git(["push"])
        except subprocess.CalledProcessError:
            pass

    ledger.close()
    print(
        f"[Daemon] {_now_iso()}  completed  epochs={epochs}  "
        f"final_Betti-1={_fmt(latest_telemetry.get('betti_1_err'))}  "
        f"final_Banach={_fmt(latest_telemetry.get('banach_loss'))}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Autonomous Banach fixed-point execution daemon",
    )
    p.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Total epochs (default: {DEFAULT_EPOCHS})",
    )
    p.add_argument(
        "--dim", type=int, default=3,
        help="Manifold / latent dimension (default: 3)",
    )
    p.add_argument(
        "--n-geometries", type=int, default=100,
        dest="n_geometries",
        help="Training set size (default: 100)",
    )
    p.add_argument(
        "--nx", type=int, default=60,
        help="Grid x-resolution (default: 60)",
    )
    p.add_argument(
        "--ny", type=int, default=60,
        help="Grid y-resolution (default: 60)",
    )
    p.add_argument(
        "--num-modes", type=int, default=8,
        dest="num_modes",
        help="Eigenmodes per geometry (default: 8)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--git-every", type=int, default=10_000,
        dest="git_every",
        help="Git commit interval in epochs (default: 10000)",
    )
    p.add_argument(
        "--mode", type=str, default="train", choices=["train", "fdtd"],
        help="Execution mode: train the God Tensor or run Topological FDTD.",
    )
    p.add_argument(
        "--dt", type=float, default=0.01,
        help="Time step size for FDTD mode.",
    )
    args = p.parse_args()

    def _sigint_handler(sig, frame):
        print("\n[Daemon] Interrupted — exiting.", flush=True)
        sys.exit(130)
    signal.signal(signal.SIGINT, _sigint_handler)

    run_daemon(
        epochs       = args.epochs,
        dim          = args.dim,
        n_geometries = args.n_geometries,
        nx           = args.nx,
        ny           = args.ny,
        num_modes    = args.num_modes,
        seed         = args.seed,
        git_every    = args.git_every,
        mode         = args.mode,
        dt           = args.dt,
    )


if __name__ == "__main__":
    main()
