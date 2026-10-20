"""Application-wide logging to data/logs/app.log.

The packaged app has no console, so anything that only goes to print()
or stderr is invisible. Everything — including uncaught exceptions in
worker threads — ends up in the rotating log file instead.
"""
import logging
import logging.handlers
import sys
import threading
from pathlib import Path

from app.core.paths import APP_LOG_FILE

_NOISY_LOGGERS = (
    "urllib3",
    "requests",
    "matplotlib",
    "numba",
    "speechbrain",
    "pytorch_lightning",
    "lightning",
    "lightning_fabric",
    "fsspec",
    "httpx",
    "httpcore",
    "huggingface_hub",
    "PIL",
    "fontTools",
    "weasyprint",
)

_configured = False


def setup_logging(level: int = logging.INFO) -> Path:
    global _configured
    if _configured:
        return APP_LOG_FILE
    _configured = True

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)

    try:
        APP_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            APP_LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError:
        pass

    stream = sys.stderr
    if stream is not None and hasattr(stream, "write") and getattr(stream, "name", "") != "nul":
        stream_handler = logging.StreamHandler(stream)
        stream_handler.setFormatter(fmt)
        root.addHandler(stream_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger(__name__).info("Logging to %s", APP_LOG_FILE)
    return APP_LOG_FILE


def install_exception_hooks() -> None:
    log = logging.getLogger("app.uncaught")

    def _excepthook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.critical("Uncaught exception", exc_info=(exc_type, exc, tb))

    def _thread_excepthook(args):
        if args.exc_type is SystemExit:
            return
        thread_name = args.thread.name if args.thread else "?"
        log.critical(
            "Uncaught exception in thread %s", thread_name, exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
        )

    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook
