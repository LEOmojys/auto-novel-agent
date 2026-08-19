from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager


class TransactionStateError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("A write transaction is already active")


@contextmanager
def write_transaction(connection: sqlite3.Connection) -> Iterator[None]:
    if connection.in_transaction:
        raise TransactionStateError
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
        connection.commit()
    finally:
        if connection.in_transaction:
            connection.rollback()


__all__ = ("TransactionStateError", "write_transaction")
