# Zero-shot prediction of missense mutation effects with ESM-2

Pipeline for scoring the effect of missense mutations using the protein language
model **ESM-2** (Meta) without fine-tuning. Validated on two homologous proteins:
**TP53** and **TP63**. We use in study these proteins because TP53 is notorious, but the most studied tumor suppressor in biology. And as for TP63 - TP53 homolog, tolerability assessment. TP63 and TP53 are paralogs—genes that arose from the duplication of a common ancestral gene. They share a similar domain structure (specifically, a DNA-binding domain) but perform different functions:

TP53 - cell cycle control, apoptosis, tumor suppression.
TP63 - epithelial development, stem cell function; it is not a tumor suppressor.
The logic behind the test is as follows: if the method detects pathogenic mutations not only in the "well-studied" TP53 but also in its relative, it implies that the method relies on shared evolutionary principles rather than on the specific, idiosyncratic features of a single protein.

## TL;DR
The zero-shot approach transfers between homologs, ROC-AUC is nearly identical
on TP53 (0.877) and TP63 (0.875), despite TP63 having a much smaller benign set.

## Method

**Masked marginal scoring** with ESM-2 (`facebook/esm2_t33_650M_UR50D`):

For each position in the protein sequence, the residue is replaced with `<mask>`,
and the model predicts probabilities for all 20 amino acids.

**Negative score** it is mean substitution is unlikely given evolutionary context (potentially pathogenic).
**Score near zero** it is mean substitution is tolerated (likely benign).

## Pipeline

1. **Full scan** (`src/mutation_scorer.py`):
   Scans all possible missense mutations in a protein.
   For TP53 (393 aa): 393 × 19 = 7,467 mutations uses the smaller `esm2_t12_35M_UR50D` for speed.

2. **Validation** (`src/validate_650m.py`, `src/validate_tp63.py`):
   Scores curated ClinVar mutations (Pathogenic / Benign, unambiguous classification). Uses the larger `esm2_t33_650M_UR50D` for accuracy. Computes ROC-AUC.

3. **Web app** (`src/app.py`):
   Streamlit interface with validation plot, heatmap, and top pathogenic mutations.

## Key findings

1. **Model size matters.** The small `t12_35M` model gives weak signal (R175H score ≈ −0.5),
   while `t33_650M` separates classes strongly (R175H score ≈ −6.0).
