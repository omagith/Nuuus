"""Shared exception types."""


class Cancelled(Exception):
    """Raised inside long-running work when the user cancels."""
