# medical_claims_analysis

# Quick Start (Poetry)

## Install Poetry
pip install poetry

## install deps
poetry config virtualenvs.in-project true
poetry env use 3.12    
poetry install
source .venv/bin/activate

## Reformat code
make pretty

## Code structure

- **/slides/** — final presentation assets
- **preprocessor.py** — data loader & basic preprocessing
- **eda.py** — quick data-quality & overview tables
- **prediction.py** — next-month drop risk per payer (logistic regression)
- **normalization.py** — WIP: normalization helpers for specialties/services
