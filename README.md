# Student At-Risk - OULAD project

## Setup (VS Code)
1. Create virtual env:
   - `python -m venv .venv`
   - Activate: Windows: `.venv\Scripts\activate`  Linux/Mac: `source .venv/bin/activate`
2. Install: `pip install -r requirements.txt`
3. Put your OULAD folder inside the project root: `student-at-risk-project/OULAD/` (with CSVs).
4. Run training: `python src/train.py OULAD`
5. Evaluate: `python src/evaluate.py models/rf_at_risk.joblib OULAD`

## Notes
- The code tries to be defensive (checks files exist).
- If your CSV column names differ slightly, inspect the CSVs and adjust mapping in `preprocess.py` and `features.py`.
- Use `notebooks/` for EDA.
