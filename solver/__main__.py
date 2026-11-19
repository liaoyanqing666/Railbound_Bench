"""Command-line entry for exact/budgeted layout search."""
import argparse
import json
from pathlib import Path
import sys

from . import solve


def main():
    parser = argparse.ArgumentParser(description='Solve a layout within a rail budget and optionally prove minimum rails.')
    parser.add_argument('level', type=Path)
    parser.add_argument('--tracks', type=int, help='Override the copied level track budget')
    parser.add_argument('--timeout', type=float, default=600.0, help='Maximum search time in seconds (default: 600)')
    parser.add_argument('--node-limit', type=int, default=0, help='Search node cap; 0 disables it (default)')
    parser.add_argument('--max-ticks', type=int, default=10000)
    parser.add_argument('--backend', choices=('auto', 'native', 'python'), default='auto',
                        help='Search backend; auto builds the optional native accelerator when a compiler is available')
    parser.add_argument('--feasible', action='store_true', help='Stop at the first verified feasible layout')
    parser.add_argument('--incumbent', type=Path, help='Optional known solution, which is checked before use')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = solve(args.level, args.tracks, timeout=args.timeout, node_limit=args.node_limit or None,
                   max_ticks=args.max_ticks, optimize=not args.feasible, incumbent=args.incumbent,
                   backend=args.backend)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    else:
        sys.stdout.write(text)
    return 0 if result['success'] else 2 if result['status'] == 'UNSAT' else 3


if __name__ == '__main__':
    raise SystemExit(main())
