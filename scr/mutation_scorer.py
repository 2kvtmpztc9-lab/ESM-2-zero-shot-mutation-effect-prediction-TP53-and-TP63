import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
import pandas as pd

# --- Загрузка модели ---
model_name = "facebook/esm2_t12_35M_UR50D"  # маленькая модель для теста
tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmForMaskedLM.from_pretrained(model_name)
model.eval()

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


def compute_mutation_score(sequence, position, mutant_aa):
    seq_list = list(sequence)
    wildtype_aa = seq_list[position]

    masked_seq = seq_list.copy()
    masked_seq[position] = "<mask>"
    masked_seq_str = "".join(masked_seq)

    inputs = tokenizer(masked_seq_str, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    token_position = position + 1  # +1 из-за <cls>
    pos_logits = logits[0, token_position, :]
    log_probs = torch.log_softmax(pos_logits, dim=-1)

    wt_token_id = tokenizer.convert_tokens_to_ids(wildtype_aa)
    mut_token_id = tokenizer.convert_tokens_to_ids(mutant_aa)

    score = (log_probs[mut_token_id] - log_probs[wt_token_id]).item()
    return score, wildtype_aa


def scan_all_mutations(sequence):
    results = []
    for pos in range(len(sequence)):
        wt = sequence[pos]
        for mut in AA_LIST:
            if mut == wt:
                continue
            score, _ = compute_mutation_score(sequence, pos, mut)
            results.append({"pos": pos + 1, "wt": wt, "mut": mut, "score": score})
    return pd.DataFrame(results)


# --- Тестовый запуск ---
if __name__ == "__main__":
    tp53_full = "MMNFETSRCATLQYCPDPYIQRFVETPAHFSWKESYYRSTMSQSTQTNEFLSPEVFQHIWDFLEQPICSVQPIDLNFVDEPSEDGATNKIEISMDCIRMQDSDLSDPMWPQYTNLGLLNSMDQQIQNGSSSTSPYNTDHAQNSVTAPSPYAQPSSTFDALSPSPAIPSNTDYPGPHSFDVSFQQSSTAKSATWTYSTELKKLYCQIAKTCPIQIKVMTPPPQGAVIRAMPVYKKAEHVTEVVKRCPNHELSREFNEGQIAPPSHLIRVEGNSHAQYVEDPITGRQSVLVPYEPPQVGTEFTTVLYNFMCNSSCVGGMNRRPILIIVTLETRDGQVLGRRCFEARICACPGRDRKADEDSIRKQQVSDSTKNGDGTKRPFRQNTHGIQMTSIKKRRSPDDELLYLPVRGRETYEMLLKIKESLELMQYLPQHTIETYRQQQQQQHQHLLQKQTSIQSPSSYGNSSPPLNKMNSMNKLPSVSQLINPQQRNALTPTTIPDGMGANIPMMGTHMPMAGDMNGLSPTQALPPPLSMPSTSHCTPPPPYPTDCSIVSFLARLGCSSCLDYFTTQGLTTIYQIEHYSMDDLASLKIPEQFRHAIWKGILDHRQLHEFSSPSHLLRTPSSASTVSVGSSETRGERVIDAVRFTLRQTISFPPRDEWNDFNFDMDARRNKQQRIKEEGE"

    print(f"Сканирую TP63 длиной {len(tp53_full)} а.к. ...")
    df = scan_all_mutations(tp53_full)
    print(f"Готово! Просканировано {len(df)} мутаций.")

    # Сохраняем ПОСЛЕ того, как df создан
    df.to_csv("tp63_scores.csv", index=False)
    print("Сохранено в tp63_scores.csv")

    print("\nТоп-10 самых 'вредных' мутаций:")
    print(df.nsmallest(10, "score").to_string(index=False))

    print("\nТоп-10 самых 'нейтральных' мутаций:")
    print(df.iloc[(df["score"]).abs().argsort()[:10]].to_string(index=False))
