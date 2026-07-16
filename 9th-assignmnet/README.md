# Week 9 — NEU Surface Defect Classification

Assignment notebook: `W9_NEU_Defect_Classification.ipynb`

## Recreating the environment

The `.venv/` folder is disposable — delete it whenever you need the disk space.
To rebuild it exactly (Python 3.10, Windows PowerShell):

```powershell
cd 9th-assignmnet
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Dataset

Not committed (see `.gitignore`). The notebook re-downloads it automatically on
first run via `kagglehub` (no Kaggle account needed) and rebuilds
`data/NEU-CLS/<class>/` from the Kaggle cache. So `data/` is also safe to delete.

## Running the notebook

Open in VSCode/Jupyter with the `.venv` kernel, or headless:

```powershell
.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace "W9_NEU_Defect_Classification.ipynb"
```

Full CPU run takes ~1 hour; per-epoch progress is written to `progress.log`.
