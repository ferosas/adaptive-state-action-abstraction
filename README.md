# Adaptive state-action abstraction via rate-distortion

This repository contains code for reproducing the tabular experiments in the
manuscript **"Adaptive state-action abstraction via rate-distortion"**.

The paper studies how a reinforcement-learning agent can solve a task through
a sequence of abstractions of increasing resolution. The guiding intuition is
that early in learning, a coarse description of the environment may be good
enough, because planning error is still large. Finer distinctions become useful
only once the agent has solved the current abstract problem well enough that
the abstraction itself is now the main source of error.

The code implements this idea for finite Markov decision processes. It builds
soft abstractions of state-action pairs using rate-distortion methods, plans in
the induced abstract problems, and refines the abstraction when a Bellman-style
learning error reaches the scale of the abstraction error.

## Main ideas

### 1. Soft state-action abstractions

Classical state abstractions group states together. This is useful, but it can
miss action symmetries: two states may require corresponding behaviours even
when the concrete action labels differ. The paper therefore abstracts
state-action pairs.

The abstraction is represented by stochastic encoders and decoders:

```text
state encoder:   nu_S(abstract_state | state)
action encoder:  nu_A(abstract_action | state, action)
state decoder:   eta_S(state | abstract_state)
action decoder:  eta_A(action | abstract_state, abstract_action)
```

Together these define an abstract MDP whose rewards and transitions are
obtained by decoding abstract pairs to concrete state-action pairs, applying
the original MDP dynamics, and encoding the next concrete state back into the
abstract state space.

In the code, the main implementation of this machinery is in:

- `code/core/abstraction.py`
- `code/core/planning.py`
- `code/core/adaptive.py`

### 2. A learning-abstraction error decomposition

The adaptive rule is motivated by a value-error certificate. In words, the
error of a grounded abstract value function can be bounded by two terms:

1. a **learning error**, measuring how well the current abstract problem has
   been solved;
2. an **abstraction error**, measuring how much control-relevant information is
   lost by the abstraction.

The paper formalizes this with a bound of the form

```text
value error <= constant * (Bellman residual + abstraction distortion).
```

The Bellman residual is the quantity that decreases as planning proceeds. The
abstraction distortion is the error floor imposed by the current resolution.
Once the residual falls to that floor, further planning at the same resolution
has diminishing returns, so the algorithm refines the abstraction.

### 3. Rate-distortion abstractions

The abstractions are built by solving a rate-distortion problem over
state-action pairs. Informally, the objective is

```text
state information
+ lambda * conditional action information
+ beta * expected control distortion.
```

Here `beta` controls the resolution:

- small `beta`: stronger pressure to compress, yielding coarser abstractions;
- large `beta`: stronger pressure to reduce distortion, yielding finer
  abstractions.

The information terms let the experiments separate how much compression comes
from states, actions, or their interaction. This is useful diagnostically: a
task may be compressible mostly because many states are equivalent, because
actions can be matched across contexts, or because both effects matter.

### 4. Adaptive refinement

The adaptive experiments use a ladder of `beta` values. The planner starts at a
coarse abstraction and monitors the grounded Bellman residual. When the
residual becomes comparable to the current abstraction distortion, the code
switches to a finer abstraction and transfers the current value estimate.

This produces a coarse-to-fine planning trajectory rather than choosing a fixed
abstraction resolution in advance.

## Experiments in this repository

The code covers four tabular domains:

1. `Four Rooms`: a navigation benchmark with spatial bottlenecks.
2. `Taxi`: a task with context-dependent relevance of pickup and dropoff
   actions.
3. `DoorKey`: a small fully observable gridworld with task-phase structure.
4. `SysAdmin`: a ring-structured maintenance benchmark used to test scaling.

The folder structure follows this order:

```text
code/exp1_four_rooms/
code/exp2_taxi/
code/exp3_doorkey/
code/exp4_sysadmin/
```

Shared code lives in:

```text
code/core/       abstraction, planning, adaptive schedules, output helpers
code/analysis/   postprocessing and plotting utilities
scripts/         end-to-end reproduction scripts
```

## Setup

Run commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The experiments use only:

```text
numpy
scipy
matplotlib
```

## A quick example

The fastest way to check the installation is to run the reduced smoke test:

```bash
python scripts/reproduce_all.py --quick --num-workers 2
```

This uses short beta schedules, fewer Blahut-Arimoto iterations, and very small
planning budgets. It verifies that the full pipeline works, but it is not meant
to reproduce the paper figures.

## Reproducing the experiments

To run all experiments with the paper-facing settings:

```bash
python scripts/reproduce_all.py --num-workers 4
```

This writes raw traces, summaries, postprocessed CSVs, and figures under
`results/`, and writes compact paper-facing CSV tables under `paper_data/`.

You can also run subsets:

```bash
# Three classic tabular benchmarks: Four Rooms, Taxi, and DoorKey.
python scripts/reproduce_all.py --experiments toy --num-workers 4

# SysAdmin scaling sweep.
python scripts/reproduce_all.py --experiments sysadmin --num-workers 4
```

The SysAdmin sweep is the slowest part of the reproduction because it computes
fixed-point metrics and uses a dense adaptive `beta` ladder for `N=2,...,7`.

## Direct experiment commands

Each domain can be run directly:

```bash
python code/exp1_four_rooms/run_four_rooms_experiment.py
python code/exp2_taxi/run_taxi_experiment.py
python code/exp3_doorkey/run_doorkey_experiment.py
python code/exp4_sysadmin/exp_scaling.py --n-min 2 --n-max 7 --eval-interval 1
```

After a direct toy run, rebuild its report CSVs and figures with the matching
`results.py` and `make_figures.py` scripts in the same experiment folder.

## Outputs

The main outputs are:

- `results/<domain>/traces.csv`: planning and adaptive-refinement traces;
- `results/<domain>/summary.json`: run configuration and final metrics;
- `results/<domain>/final_results/`: postprocessed CSVs used for figures;
- `results/<domain>/figures/`: diagnostic PNG/PDF figures;
- `paper_data/`: compact CSVs collecting the quantities plotted in the paper.

## Notes

This repository is intended as a compact reproduction bundle. It omits local
working files, large caches, exploratory experiments, and paper build artifacts.

## Citation

If you use this code, please cite the companion paper:

```bibtex
@misc{rosas2026adaptive,
  title  = {Adaptive state-action abstraction via rate-distortion},
  author = {Rosas, Fernando E.},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## License

This code is released under the MIT License. See `LICENSE` for details.
