Easy(EZ) (P)air (D)istribution (F)unction & <br>Easy(EZ) (P)air distribution function f(IT)
===========================================
**EZPDF** provides S(q), F(q) and G(r), and Compton scattering from experimental I(q) data and composition, while also generating I(q), S(q), F(q), and G(r) from .xyz model files.
<br><br>**EZPIT** is a simulation-based structural refinement software developed on the foundation of the EZPDF engine.
Its primary objective is to optimize atomic models by calculating theoretical scattering signals and fitting them to experimental Pair Distribution Function (PDF) data.

## Install

Requires Python 3.12+.

```bash
pip install ezpit          # core (numpy/scipy) only
pip install ezpit[gui]     # + Qt GUI (PySide6, pyqtgraph, pandas)
pip install ezpit[examples]  # + matplotlib, for running examples/
```

From source with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra gui --extra examples
```

## Usage

```bash
ezpdf-gui                                    # launch the Qt GUI
ezpdf-wh-smooth <file>                       # Whittaker-Henderson smoothing CLI
uv run --extra examples python examples/demo_A_experimental_data.py  # experimental I(q) -> S(q)/F(q)/G(r)
```

See `examples/` for scripted workflows (experimental data, .xyz models, Compton scattering) and [docs/tutorial.md](docs/tutorial.md) for the full walkthrough.

## EZPDF/EZPIT Contributors

**Author:**
<br> - Gihan Kwon

**Contributors:**
<br>*Brookhaven National Laboratory*
<br> - Tj Barrett
<br> - Cheng-Hung Lin
<br> - Joshua Lynch
<br> - Ajith Pattammattel
<br> - Nghia Vo
<br> - Jakub Wlodek
<br> - Hui Zhong
<br>*Northwestern University*
<br> - Dustin Zhao
<br>*Stony Brook University*
<br> - Dongyoon Lee
<br> - Yichen Liu
