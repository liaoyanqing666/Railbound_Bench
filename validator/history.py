"""Selection of feedback frames without discarding the complete simulation trace."""
from copy import deepcopy


def select_feedback(trace, history_steps=None, history_gap=1, anchor=None):
    """Sample backwards from an index, then return chronological snapshots.

    ``None`` means as many samples as fit through the anchor.
    If fewer than the requested samples fit, include state zero when absent.
    A loop caller supplies the index of the first occurrence as the anchor.
    """
    if type(history_gap) is not int or history_gap < 1:
        raise ValueError('history_gap must be a positive integer')
    if history_steps is not None and (type(history_steps) is not int or history_steps < 1):
        raise ValueError('history_steps must be a positive integer or None')
    if not trace:
        return []
    if anchor is None:
        anchor = len(trace) - 1
    if type(anchor) is not int or not 0 <= anchor < len(trace):
        raise ValueError('anchor must index a trace state')
    indices = []
    index = anchor
    while index >= 0 and (history_steps is None or len(indices) < history_steps):
        indices.append(index)
        index -= history_gap
    if (history_steps is None or len(indices) < history_steps) and indices[-1] != 0:
        indices.append(0)
    return [deepcopy(trace[index]) for index in reversed(indices)]
