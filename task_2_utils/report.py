"""Text formatting for the report scripts.

Only presentation lives here. It exists so that `task_2.py` and
`batch_task_2.py` can each print a section with a single ``print`` call
instead of a loop of them, and so that column widths are not repeated as magic
numbers next to every table.
"""

from __future__ import annotations

from typing import Iterable, Sequence

__all__ = ["table"]


def table(
    header: Sequence[str],
    rows: Iterable[Sequence],
    indent: str = "    ",
    align: str | None = None,
) -> str:
    """A fixed-width table as one string (no trailing newline).

    ``align`` is one character per column, ``l`` or ``r``; the default is the
    usual "labels left, numbers right". Column widths come from the content,
    so a wider count never breaks the layout.
    """
    head = [str(h) for h in header]
    cells = [[str(c) for c in row] for row in rows]
    align = align or "l" + "r" * (len(head) - 1)
    widths = [
        max([len(head[i])] + [len(row[i]) for row in cells])
        for i in range(len(head))
    ]

    def line(values: Sequence[str]) -> str:
        return indent + "  ".join(
            value.ljust(width) if how == "l" else value.rjust(width)
            for value, width, how in zip(values, widths, align)
        ).rstrip()

    return "\n".join([line(head)] + [line(row) for row in cells])
