# Zero-shot prediction of missense mutation effects with ESM-2

Pipeline for scoring the effect of missense mutations using the protein language
model **ESM-2** (Meta) without fine-tuning. Validated on two homologous proteins:
**TP53** and **TP63**.

## TL;DR

| Protein | N mutations | Mean score (pathogenic) | Mean score (benign) | Gap | ROC-AUC |
|---|---|---|---|---|---|
| TP53 | 34 | −6.54 | −1.43 | 5.11 | **0.877** |
| TP63 | 20 | −7.28 | −0.01 | 7.27 | **0.875** |

![Comparison TP53 vs TP63](figures/comparison_tp53_tp63.png)

The zero-shot approach transfers between homologs: ROC-AUC is nearly identical
on TP53 (0.877) and TP63 (0.875), despite TP63 having a much smaller benign set.

## Method

**Masked marginal scoring** with ESM-2 (`facebook/esm2_t33_650M_UR50D`):

For each position in the protein sequence, the residue is replaced with `<mask>`,
and the model predicts probabilities for all 20 amino acids.

- **Negative score** → substitution is unlikely given evolutionary context (potentially pathogenic).
- **Score near zero** → substitution is tolerated (likely benign).

## Pipeline

1. **Full scan** (`src/mutation_scorer.py`):
   - Scans all possible missense mutations in a protein.
   - For TP53 (393 aa): 393 × 19 = **7,467 mutations**.
   - Uses the smaller `esm2_t12_35M_UR50D` for speed.

2. **Validation** (`src/validate_650m.py`, `src/validate_tp63.py`):
   - Scores curated ClinVar mutations (Pathogenic / Benign, unambiguous classification).
   - Uses the larger `esm2_t33_650M_UR50D` for accuracy.
   - Computes ROC-AUC.

3. **Web app** (`src/app.py`):
   - Streamlit interface with validation plot, heatmap, and top pathogenic mutations.

## Key findings

1. **Model size matters.** The small `t12_35M` model gives weak signal (R175H score ≈ −0.5),
   while `t33_650M` separates classes strongly (R175H score ≈ −6.0).

2. **Method transfers between homologs.** ROC-AUC ≈ 0.88 on both TP53 and TP63.

3. **Conservative domains are easier.** Hot-spot mutations in the DNA-binding
   domain of TP53 (R248W, C176F, R282W) get scores −8…−12.
   Mutations in the C-terminal regulatory region (positions 335–347) are harder —
   both false positives (R342P, R335L) and false negatives (R337L, R337H).

4. **Usage** (the initiator Met is often cleaved in the mature protein)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
```
```
cd src
python mutation_scorer.py
```

```
python validate_650m.py          # TP53
python validate_tp63.py          # TP63
```

```
streamlit run app.py
```


