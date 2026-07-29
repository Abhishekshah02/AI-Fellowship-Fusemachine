# 12th Assignment — End-to-End NER for Customer Support Ticket Analysis

Single deliverable: [Assignment3_NER_CustomerSupport.ipynb](Assignment3_NER_CustomerSupport.ipynb)

## Contents
- **Task 1** — Preprocessing + EDA on CoNLL-2003 (sentence counts, vocabulary size, entity frequencies, most/least frequent words)
- **Task 2** — Word-level + contextual feature engineering for sequence labeling
- **Task 3** — CRF model (sklearn-crfsuite), precision/recall/F1 and entity-wise breakdown
- **Task 4** — Error analysis with sample predictions and top-3 error patterns
- **Task 5** — Business impact discussion + recommendation
- **Task 6** — Same pipeline on WNUT-17 emerging entities, with CoNLL vs WNUT comparison

## Environment
Reuses the existing virtual environment at `../10th-assignment/.venv` (Python 3.10).
Extra packages for this assignment were installed into it — see `requirements.txt`.

To run the notebook, select that interpreter as the Jupyter kernel, or execute headless:

```
../10th-assignment/.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace Assignment3_NER_CustomerSupport.ipynb
```

## Datasets (downloaded automatically via Hugging Face `datasets`)
- CoNLL-2003: `eriktks/conll2003` (parquet revision)
- WNUT-17: `leondz/wnut_17` (`refs/convert/parquet` revision — original loading script is deprecated in datasets v3+)
