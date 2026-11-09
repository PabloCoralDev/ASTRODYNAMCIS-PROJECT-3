# Project 3 - how to run this

For TA: Quick guide on how to run the code:

## Setup (if on a MAC / if not it's slightly different)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (from this folder)

```bash
python tasks1to3.py   # Tasks 1-3: Kepler propagation, groundtracks, hemisphere/crossing-point/apoapsis analysis
```

Figures are saved to `report/figures/`. All other `.py` files only hold the functions used by `tasks1to3.py`.

Note: the cover page of `report/main.tex` references `report/figures/honor_code.png` (signed honor code
photo) -- that file isn't in this repo yet, drop it in before building the PDF.
