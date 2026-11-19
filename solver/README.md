# Layout solver

[Project guide / 项目说明](../README.md)

The solver places ordinary rails, junctions and legal semaphores on the original
board within its budgets. Original rails, mechanisms, cars, IDs and dimensions
remain fixed. Every answer returned by the C++ backend is replayed through the
independent Python validator, including its immutable-content and budget checks.

## Usage

```powershell
python -m solver levels/extended/easy/E-CT0022.json --timeout 600 --node-limit 0 --max-ticks 10000 --output temp/solve.json
python -m solver levels/extended/easy/E-BF0013.json --feasible --timeout 600 --node-limit 0 --max-ticks 10000
```

```python
from pathlib import Path
from solver import solve

result = solve(Path("levels/extended/easy/E-CT0022.json"),
               timeout=600, node_limit=None, max_ticks=10000, optimize=True)
print(result["status"], result["tracks_used"], result["minimum_tracks"])
answer = result["answer"]       # add_tracks and optional add_semaphores.
solution = result["solution"]   # Expanded full board for local tools.
```

By default, search minimizes new track cells with a 600-second allowance, no
search-node cap and 10,000 ticks per simulation. `--feasible` stops at the first
checked solution. `--tracks` (`track_budget` in Python) changes the budget in a
copy of the puzzle; the returned full solution carries that override. The solver
keeps the original board dimensions and semaphore allowance.

An optional `--incumbent` / `incumbent` supplies a known additions answer, such
as an annotated reference file or a previous result's `answer`. The solver validates it before
use. To measure difficulty, start without an existing answer: checking a supplied
solution can prove a minimum immediately and does not measure independent solving.

## Results and limits

| Status | Meaning |
| --- | --- |
| `OPTIMAL` | A checked solution exists and a lower bound or complete search proves that no layout uses fewer new rail cells. |
| `SAT` | A checked solution exists; the minimum has not been proved, or only feasibility was requested. |
| `UNSAT` | A valid bound or complete search proves no solution within the stated track and semaphore budgets. |
| `UNKNOWN` | A time, node or simulation limit prevented a conclusion and no checked solution was found. |

`minimum_tracks` is populated only for `OPTIMAL`. `tracks_used` gives the found
answer's cost; `lower_bound` gives the number of rails every solution must need.
If a checked answer meets that bound, Python returns `reason='lower_bound_matched'`
and `backend='python'` before launching C++. Otherwise the selected backend
searches for an answer or a cheaper layout. `backend` records what ran and
`reason` explains why it stopped.

Optimization counts new rail cells. It keeps the semaphore budget fixed and does
not minimize signals or execution length. `simulation_steps` records the
successful execution's movement ticks, including waits.

Timeouts preserve a previously checked answer as `SAT`; otherwise they return
`UNKNOWN`. A simulator `resource_limit` also leaves the affected branch unresolved.
`statistics.search_exhausted` records actual search-tree exhaustion;
`statistics.complete_search` also permits a proof from matching a lower bound.
Elapsed time is checked between search nodes or simulator calls. `max_ticks`
limits each complete execution, including reused prefixes. In Python, set
`timeout` or `node_limit` to `None` to disable that limit; `--node-limit 0`
disables the CLI node cap.

The corpus search protocol uses 600 seconds, no node cap and 10,000 ticks per
simulation, starting without a reference and stopping at the first feasible
answer. It solved 229 of 233 official puzzles and 992 of 1,000 extension puzzles.
The remaining puzzles have validated reference answers and `UNKNOWN` independent
search records. Extension budgets have 987 proved minima (970 from C++ search,
17 from Python lower-bound checks) and 13 checked feasible `SAT` records.
See the [extension inventory](../levels/extended/manifest.json).

The separate 200-puzzle hard set has checked reference answers. Its saved
optimization records label 136 answers `OPTIMAL` and 64 `SAT`; see the
[puzzle-set description](../README.md#puzzle-sets) and
[search records](../solutions/extended-hard/statistics.json) for the limits,
bounds and stopping reasons. Difficulty checks may stop at 100,001 nodes;
minimum-track searches use the full 600-second allowance without a node cap.

## Backends

`backend='auto'` builds the adjacent C++14 source when `g++` or `clang++` is on
`PATH`, caching the executable in the operating system's temporary directory
outside the checkout. Otherwise it selects the portable Python search.
`backend='native'` requires a compiler; `backend='python'` explicitly selects
Python. The CLI accepts the same choices through `--backend`. No external solver
package is required. Prebuild before measuring search time to exclude the first
compilation:

```powershell
python -B -c "from solver.native import executable; print(executable())"
```

Python search uses the validator's transition engine. C++ implements the same
synchronous rules independently in compact arrays and receives a final public
Python replay before returning an answer. Both stop before the earliest
undecided move. Normal validator calls still retain every animation frame.

## Search and proof

Search combines connectivity checks with synchronous vehicle simulation:

1. An optimistic, undirected graph ignores directions and gate timing. It finds
   mandatory empty cells and a lower bound on the new rails needed to connect
   required cars, their stops and compatible locomotives.
2. A directed reachability check rejects a partial layout when a required car
   cannot reach an unserved stop or matching locomotive. It allows every
   remaining rail orientation, both switching states, and ignores gates, timing
   and collisions, so it only rejects impossible connections.
3. Search chooses the first rail or possible semaphore a vehicle can inspect,
   then simulates the resulting moves. Decoy wreck choices remain available.
   A new signal also receives any adjoining junction required for legal placement,
   including a junction that no vehicle traverses.
4. Search rejects branches whose fixed moves fail, whose track budget is used up,
   or whose lower bound cannot improve an existing checked answer. Unused rails
   need not be enumerated because they cannot change an earlier entry event;
   static signal connections are handled explicitly.

A directed distance field orders choices: fixed rails and tunnels cost zero,
and entering an original empty cell costs one. It respects tunnel mouths and
locomotive approach directions while allowing both switching states. This ordering
does not establish a track lower bound because a route may reuse a cell in several
directions. Decoy wrecks and signal choices also receive ordering preferences;
these preferences retain every branch needed for a proof.

Three initial passes cap repeated visits to required-car positions and headings
at one, two and three, each with at most 100,000 nodes. These passes are incomplete
ways to find an answer. If they fail, search restarts without visit caps.
`heuristic_passes` and `heuristic_search_nodes` record their work. Only a valid
bound or complete search proves `UNSAT` or a minimum.

C++ reuses complete simulation snapshots immediately before an undecided tick
when the observed prefix is unchanged. Snapshots include dynamic rails,
mechanisms, vehicles, waits, deliveries, queued events, gate closures and loop
state. Adding a signal or changing rail geometry observed by an existing signal
replays from tick zero so earlier signal openings are included. Loop detection
uses every changing value that can affect future movement.

Rail choices can remain a set of codes with identical observed entry behavior;
later approaches split that set as needed. Successful layouts choose concrete
tiles. Signal release depends on physical junction ports, so signal-connected
cells retain concrete geometry. Python uses concrete choices throughout when new
signals are allowed. C++ retains rail sets until a signal is placed, then fixes
that signal's rail and every assigned or future port-connected neighbor.

The optimizing search tightens its upper bound whenever it finds a cheaper
answer. A checked B-rail answer plus a proof that B-1 rails are infeasible proves
the minimum. A timeout at B-1 leaves B as an upper bound. Lowering budgets only
advances a proven lower bound when the result is `UNSAT`.

## Workload statistics

| Field | Meaning |
| --- | --- |
| `elapsed_seconds`, `cpu_seconds` | Wall time and CPU time; native subprocess CPU is included. Windows uses `GetProcessTimes`. |
| `search_nodes`, `simulations` | Search choices visited and simulation executions. |
| `simulation_ticks` | Transitions actually computed, including final answer verification. |
| `logical_simulation_ticks`, `reused_prefix_ticks` | Equivalent full-replay work and the transitions saved by snapshot reuse. |
| `checkpoint_resumes`, `full_replays` | Native executions resumed from a snapshot or begun at tick zero. |
| `witness_validations`, `incumbent_validations` | Checks of found answers and user-supplied answers, respectively. |

Additional counters record bounds, rejected candidates and resource limits. Bounds
are cached by occupied and explicitly empty cells; the optimistic bound ignores
rail orientation. Record limits and workload counters with timings. They measure
this solver's work, rather than human or model difficulty.

## References

Algorithm references include FoxtrotOnce's
[C++ search](https://github.com/FoxtrotOnce/RailboundSolver/tree/main/cpp) and the
[MiniZinc Railbound article](https://thinhcorner.com/writing/railbound-constraint-solver).
Status names follow the conventions documented by
[OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver/).
The solver builds on code from FoxtrotOnce's
[RailboundSolver](https://github.com/FoxtrotOnce/RailboundSolver), adapted to this
benchmark's rules; no external solver binaries are bundled or required. That
repository is MIT licensed, and its notice is retained below.

```text
MIT License

Copyright (c) 2025 FoxtrotOnce

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
