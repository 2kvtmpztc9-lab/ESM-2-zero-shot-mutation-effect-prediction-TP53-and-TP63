import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
import torch
import pandas as pd
from transformers import EsmTokenizer, EsmForMaskedLM

model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmForMaskedLM.from_pretrained(model_name)
model.eval()

tp53_full = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"

def score_mutation(sequence, position, mutant_aa):
    seq_list = list(sequence)
    wt = seq_list[position - 1]
    seq_list[position - 1] = "<mask>"
    inputs = tokenizer("".join(seq_list), return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    log_probs = torch.log_softmax(logits[0, position, :], dim=-1)
    wt_id = tokenizer.convert_tokens_to_ids(wt)
    mut_id = tokenizer.convert_tokens_to_ids(mutant_aa)
    return (log_probs[mut_id] - log_probs[wt_id]).item()

val = pd.read_csv(os.path.join(DATA, "validation_tp53.csv"))
val["score"] = val.apply(
    lambda r: score_mutation(tp53_full, int(r["pos"]), r["mut"]), axis=1
)
val.to_csv(os.path.join(DATA, "validation_tp53_scored.csv"), index=False)
print(val.to_string(index=False))
print("\nСредний score патогенных:", round(val[val.label=="pathogenic"].score.mean(), 2))
print("Средний score доброкачественных:", round(val[val.label=="benign"].score.mean(), 2))
