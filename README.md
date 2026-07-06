# qec-ai-decoder

**Transformer-based neural decoding for rotated surface codes under circuit-level noise, benchmarked against MWPM with a dual-baseline (blind vs. aware) evaluation protocol.**

This repository accompanies an MSc thesis at Hacettepe University and the paper
*"Data Efficiency Bottlenecks in Transformer-Based Surface-Code Decoders: A Systematic Comparison with MWPM"*.
All experiments were run on the TRUBA national HPC infrastructure (TÜBİTAK ULAKBİM) using SLURM job arrays.

Rather than claiming "AI beats MWPM", this work systematically characterizes **when and why it does not**:
the central empirical finding is a **data efficiency bottleneck** — the Transformer approaches MWPM
performance at distance *d* = 3, but fails to recover threshold scaling at *d* = 5 and *d* = 7 within a
fixed training budget of ~10⁵ shots. A power-law fit (LER ∝ N⁻⁰·⁴⁰) extrapolates to roughly
3.4 × 10⁶ training shots needed to match MWPM at *d* = 5.

---

## Contents

- [qec-ai-decoder](#qec-ai-decoder)
  - [Contents](#contents)
  - [Highlights](#highlights)
  - [The dual-baseline protocol](#the-dual-baseline-protocol)
  - [Project evolution (v1 → v3)](#project-evolution-v1--v3)
  - [Repository layout](#repository-layout)
  - [Installation](#installation)
  - [Quickstart](#quickstart)
  - [Running on an HPC cluster (TRUBA/SLURM)](#running-on-an-hpc-cluster-trubaslurm)
  - [Reproducibility](#reproducibility)
  - [Data and trained models](#data-and-trained-models)
  - [Acknowledgments](#acknowledgments)
  - [License](#license)

---

## Highlights

- **Full reproducible pipeline**: Stim circuit construction → syndrome sampling → PyTorch training →
  logical error rate (LER) evaluation against a PyMatching MWPM reference, orchestrated end-to-end
  with SLURM job arrays.
- **Dual-baseline MWPM protocol** (`evaluate_v3_blind.py`): every learned decoder is compared against
  *two* classical references — a "blind" MWPM and an "informed" MWPM — which separates *what the model
  learned about noise correlations* from *raw architectural capacity*.
- **Explicit correlated-noise injection**: nearest-neighbor `DEPOLARIZE2(p_corr)` channels are injected
  into the Stim circuit on top of the standard i.i.d. circuit-level noise model, with the injection
  methodology fully documented in code.
- **Three model generations kept side by side** (see below), so every intermediate result in the thesis
  can be regenerated.
- **Statistical care**: LER estimates carry binomial standard errors; train/val/test splits use disjoint,
  deterministic seeds recorded in a manifest.

## The dual-baseline protocol

All three decoders are evaluated on the *same* correlated test data:

| Decoder | Noise model it assumes | Role |
|---|---|---|
| `mwpm_naive` | DEM extracted from the **uncorrelated** (i.i.d.) circuit at `p_iid`; assumes `p_corr = 0` | Realistic deployment baseline — a hardware engineer calibrates MWPM without knowing the correlation structure |
| `mwpm_aware` | DEM extracted from the **correlated** circuit (`p_iid`, `p_corr`) | Informed classical ceiling — correlation is built into the matching graph, but MWPM is still limited to pairwise edge decomposition |
| `transformer` | None — learns the syndrome → logical-flip mapping directly from correlated data | The learned decoder under study |

The Transformer's position between the two references determines the honest claim:

- **≈ `mwpm_naive`** → the model has hit a *data wall* (learned no usable correlation structure),
- **between the two** → *partial learning* of the correlation structure,
- **≈ or below `mwpm_aware`** → learning is equivalent to (or exceeds) handing MWPM the hidden noise model.

Reporting only "Transformer vs. one MWPM" is misleading; the dual baseline is itself a methodological
contribution of this work.

## Project evolution (v1 → v3)

| Gen | Input representation | Model | Noise model | Purpose |
|---|---|---|---|---|
| **v1** | Flat syndrome vector (`num_detectors`,) | MLP / Conv1D | i.i.d. circuit-level (single `p`) | Baseline; shows flat MLPs do not scale with distance |
| **v2** | 3D spatiotemporal tensor `(T, H, W)` | Transformer (~550K params): learned positional encoding, `[CLS]` token, 4× pre-norm encoder layers, 8-head attention | i.i.d. circuit-level | Structure-preserving learned decoder |
| **v3** | Same as v2 | Same as v2 | i.i.d. **+ correlated** nearest-neighbor `DEPOLARIZE2(p_corr)` | Correlated regime + blind/aware MWPM evaluation |

The three generations share the same physics layer (`qec_common*.py`), the same "one expert model per
noise setting" training design, and the same LER-based evaluation, so results are directly comparable.

## Repository layout

```
qec-ai-decoder/
├── configs/
│   ├── config.yaml               # v1/v2 grid: distances × noise levels (i.i.d.)
│   ├── config_correlated.yaml    # v3 grid: distances × (p_iid, p_corr) pairs
│   └── smoke_test.yaml           # tiny end-to-end test configuration
├── slurm/
│   ├── 00_debug_test.slurm       # full pipeline on the short debug queue
│   ├── 01_generate_data.slurm    # data generation (CPU job array)
│   ├── 02_train.slurm            # v1 training (GPU job array)
│   ├── 03_evaluate.slurm         # v1 evaluation + plots
│   ├── 05_train_correlated.slurm # v3 training over (d, p_iid, p_corr) grid
│   ├── setup_env.sh              # one-time environment setup on the login node
│   └── run_all.sh                # submits the whole pipeline with dependencies
├── src/
│   ├── qec_common.py             # physics layer: rotated surface code + i.i.d. circuit-level noise (Stim)
│   ├── qec_common_v3.py          # v3: correlated-noise injection via DEPOLARIZE2 on neighbor pairs
│   ├── generate_data.py          # v1 dataset generation → compressed .npz + manifest.json
│   ├── generate_data_v3.py       # v3 dataset generation, parameterized by (p_iid, p_corr)
│   ├── dataset.py                # v1: flat syndrome vectors
│   ├── dataset_v2.py             # v2/v3: reshapes detectors into (T, H, W) tensors
│   ├── models.py                 # v1: MLPDecoder, Conv1dDecoder
│   ├── models_v2.py              # v2/v3: TransformerDecoder
│   ├── train.py                  # v1 training loop (BCE + pos_weight, early stopping)
│   ├── train_v3.py               # v2/v3 training loop
│   ├── evaluate.py               # v1 evaluation: AI vs. single MWPM reference
│   ├── evaluate_v3_blind.py      # three-way comparison: mwpm_naive / mwpm_aware / transformer
│   ├── baseline_mwpm.py          # MWPM reference decoder (Stim DEM → PyMatching)
│   └── list_tasks.py             # maps the config grid to SLURM job-array indices
├── results/                      # summary tables (CSV/JSON) and figures (small artifacts)
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

> One-off helper scripts prefixed with `_` (e.g. environment workarounds for cluster-specific PyTorch
> issues) may also be present in `src/`; they are documented in their own docstrings. Adjust this tree
> to match the final file set you publish.

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/<username>/qec-ai-decoder.git
cd qec-ai-decoder
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Install PyTorch separately for your platform/CUDA version:
# https://pytorch.org/get-started/locally/
```

Core dependencies: [`stim`](https://github.com/quantumlib/Stim) (fast stabilizer-circuit simulation),
[`pymatching`](https://github.com/oscarhiggott/PyMatching) (MWPM), `torch`, `numpy`, `pyyaml`, `matplotlib`.

## Quickstart

Run a tiny end-to-end sanity check on a laptop (CPU, a few minutes):

```bash
python src/generate_data.py --config configs/smoke_test.yaml
python src/train.py        --config configs/smoke_test.yaml --distance 3 --p 0.0100
python src/evaluate.py     --config configs/smoke_test.yaml
```

v1 pipeline for a single (d, p) point:

```bash
python src/generate_data.py --config configs/config.yaml --distance 3 --p 0.01
python src/train.py         --config configs/config.yaml --distance 3 --p 0.01
python src/evaluate.py      --config configs/config.yaml
```

v3 correlated-noise experiments (Transformer + dual MWPM baselines):

```bash
python src/generate_data_v3.py   --config configs/config_correlated.yaml
python src/train_v3.py           --config configs/config_correlated.yaml \
                                 --distance 3 --p_iid 0.005 --p_corr 0.005
python src/evaluate_v3_blind.py  --config configs/config_correlated.yaml
```

See each module's docstring for the full CLI. Outputs (summary tables, LER-vs-noise and
AI-vs-MWPM plots) are written to `results/`.

## Running on an HPC cluster (TRUBA/SLURM)

The experiment grid is *embarrassingly parallel*: every (distance, noise) pair is an independent
job. `run_all.sh` submits the three stages with SLURM dependencies so the pipeline advances on its own:

```bash
bash slurm/setup_env.sh     # one-time environment setup on the login node
bash slurm/run_all.sh       # data generation → training → evaluation, chained via --dependency=afterok
```

Notes for TRUBA users:

- Set your account name in the `#SBATCH -A` lines of the `slurm/*.slurm` scripts.
- Module names on TRUBA change over time; check `module avail` and update the `module load` lines in
  `setup_env.sh` and the SLURM scripts accordingly. Heavy packages (PyTorch, CUDA) should come from the
  centrally provided modules; only lightweight QEC-specific packages (`stim`, `pymatching`) are pip-installed
  into a `--system-site-packages` venv, in line with TRUBA's filesystem guidance.
- Data generation is a CPU job; only training and evaluation use the GPU queues.

The scripts are written for TRUBA but the pattern (config grid → `list_tasks.py` → job array) ports
directly to any SLURM cluster.

## Reproducibility

- **Deterministic seeds**: every dataset file gets a seed derived from `(split, distance, p)` with fixed
  offsets, so train/val/test are disjoint by construction (no data leakage) and every file can be
  regenerated bit-for-bit by Stim.
- **Manifest-driven**: `generate_data*.py` writes a `manifest.json` indexing every dataset; training and
  evaluation resolve data exclusively through it.
- **Fair comparison design**: one expert model per (d, noise) setting — mirroring how MWPM is rebuilt with
  the matching noise model at each grid point.
- **Uncertainty reporting**: all LER values carry binomial standard errors, `sqrt(p(1-p)/N)`.
- **Caveat**: exact bitwise reproduction of *training* is limited by GPU nondeterminism (cuDNN kernels);
  dataset generation and MWPM evaluation are fully deterministic.

## Data and trained models

Generated datasets (`*.npz`) and model checkpoints (`*.pt`) are **not** tracked in git (see `.gitignore`) —
they total several GB and are fully regenerable from the configs and seeds above.

## Acknowledgments

The numerical calculations reported in this work were performed at TÜBİTAK ULAKBİM,
High Performance and Grid Computing Center (TRUBA resources).

This work builds on the excellent open-source tools [Stim](https://github.com/quantumlib/Stim) and
[PyMatching](https://github.com/oscarhiggott/PyMatching).

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
