# from rouge import Rouge

# def compute_rouge_scores(hypothesis, reference):
#     rouge = Rouge()
#     scores = rouge.get_scores(hypothesis, reference, avg=True)

#     # Extracting F1 scores
#     rouge_1_f1 = scores['rouge-1']['f']
#     rouge_2_f1 = scores['rouge-2']['f']
#     rouge_l_f1 = scores['rouge-l']['f']

#     return rouge_1_f1, rouge_2_f1, rouge_l_f1

# import torch
# from bert_score import score

# def compute_bertscore(candidate, reference):
#     P, R, F1 = score([candidate], [reference], lang="en", model_type="bert-base-uncased", device="cuda" if torch.cuda.is_available() else "cpu")
#     return P.item(), R.item(), F1.item()


# import nltk
# from nltk.translate.bleu_score import sentence_bleu
# from nltk.tokenize import word_tokenize

# def compute_bleu(candidate,reference):
#     reference_tokenized = [word_tokenize(reference)]
#     candidate_tokenized = word_tokenize(candidate)
#     return sentence_bleu(reference_tokenized, candidate_tokenized)



import os
import pandas as pd
import torch
import nltk
from rouge import Rouge
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# Download necessary NLTK tokenizers if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def compute_rouge_scores(hypothesis, reference):
    if not hypothesis.strip() or not reference.strip():
        return 0.0, 0.0, 0.0
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference, avg=True)
    return scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']

def compute_bertscore(candidate, reference):
    if not candidate.strip() or not reference.strip():
        return 0.0, 0.0, 0.0
    # Forces standard CPU computation if Mac GPU frameworks aren't explicitly bound
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P, R, F1 = score([candidate], [reference], lang="en", model_type="distilbert-base-uncased", device=device, verbose=False)
    return P.item(), R.item(), F1.item()

def compute_bleu(candidate, reference):
    if not candidate.strip() or not reference.strip():
        return 0.0
    reference_tokenized = [word_tokenize(reference)]
    candidate_tokenized = word_tokenize(candidate)
    # Uses smoothers to prevent zero-division drops on short sentences
    chencherry = SmoothingFunction()
    return sentence_bleu(reference_tokenized, candidate_tokenized, smoothing_function=chencherry.method1)

# =====================================================================
# AUTOMATED RUNNER ENGINE
# =====================================================================
if __name__ == '__main__':
    print("=============================================")
    print("   WALERT NLG TEXT METRICS GENERATOR         ")
    print("=============================================\n")

    DATA_FILE = "../../data/walert_intent_results.csv"

    if not os.path.exists(DATA_FILE):
        print(f"Error: Target data file missing at {DATA_FILE}")
        print("Creating mock baseline evaluations for standalone validation...")
        # Generates a clean baseline preview if tracking tables vary across folders
        r1, r2, rl = compute_rouge_scores("RMIT Software Engineering has a placement", "RMIT Software Engineering includes a mandatory placement year")
        bp, br, bf = compute_bertscore("RMIT Software Engineering has a placement", "RMIT Software Engineering includes a mandatory placement year")
        bleu_score = compute_bleu("RMIT Software Engineering has a placement", "RMIT Software Engineering includes a mandatory placement year")
        
        print("\n---------------------------------------------")
        print(" SAMPLE METRIC MATRIX PREVIEW                ")
        print("---------------------------------------------")
        print(f" ROUGE-1 F1-Score         | {r1:.4f}")
        print(f" ROUGE-2 F1-Score         | {r2:.4f}")
        print(f" ROUGE-L F1-Score         | {rl:.4f}")
        print(f" BLEU Score               | {bleu_score:.4f}")
        print(f" BERTScore F1-Semantic    | {bf:.4f}")
        print("---------------------------------------------\n")
    # else:
    #     print(f"Processing data logs from {DATA_FILE}...")
    #     df = pd.read_csv(DATA_FILE)
        
    #     # Maps text parameters safely regardless of case strings
    #     cand_col = [c for c in df.columns if 'predict' in c.lower() or 'response' in c.lower() or 'phrase' in c.lower()][0]
    #     ref_col = [c for c in df.columns if 'true' in c.lower() or 'ground' in c.lower() or 'intent' in c.lower()][0]

    #     rouge1_list, rouge2_list, rougel_list, bleu_list, bert_list = [], [], [], [], []
        
    #     print(f"Evaluating {len(df)} generation response pairs...")
    #     for _, row in df.iterrows():
    #         cand = str(row[cand_col])
    #         ref = str(row[ref_col])
            
    #         r1, r2, rl = compute_rouge_scores(cand, ref)
    #         _, _, bf = compute_bertscore(cand, ref)
    #         bl = compute_bleu(cand, ref)
            
    #         rouge1_list.append(r1)
    #         rouge2_list.append(r2)
    #         rougel_list.append(rl)
    #         bert_list.append(bf)
    #         bleu_list.append(bl)

    #     print("\n---------------------------------------------")
    #     print(" FINAL SYSTEM GENERATION ASSESSMENT MATRIX  ")
    #     print("---------------------------------------------")
    #     print(f" Mean ROUGE-1 F1-Score    | {sum(rouge1_list)/len(rouge1_list):.4f}")
    #     print(f" Mean ROUGE-2 F1-Score    | {sum(rouge2_list)/len(rouge2_list):.4f}")
    #     print(f" Mean ROUGE-L F1-Score    | {sum(rougel_list)/len(rougel_list):.4f}")
    #     print(f" Mean BLEU Textual Score  | {sum(bleu_list)/len(bleu_list):.4f}")
    #     print(f" Mean BERTScore F1-Value  | {sum(bert_list)/len(bert_list):.4f}")
    #     print("---------------------------------------------\n")

    else:
        print(f"Processing data logs from {DATA_FILE}...")
        df = pd.read_csv(DATA_FILE)
        
        # Print column names so we can see exactly what the CSV holds
        print(f"Detected columns in CSV: {list(df.columns)}")
        
        # Dynamically map columns or fall back to column 0 and column 1
        cand_col = df.columns[0] # Fallback default
        ref_col = df.columns[1]  # Fallback default
        
        for col in df.columns:
            if 'predict' in col.lower() or 'response' in col.lower() or 'phrase' in col.lower():
                cand_col = col
            if 'true' in col.lower() or 'ground' in col.lower() or 'intent' in col.lower():
                ref_col = col
                
        print(f"Using candidate text column: '{cand_col}'")
        print(f"Using reference text column: '{ref_col}'\n")

        rouge1_list, rouge2_list, rougel_list, bleu_list, bert_list = [], [], [], [], []
        
        print(f"Evaluating {len(df)} generation response pairs...")
        for _, row in df.iterrows():
            cand = str(row[cand_col])
            ref = str(row[ref_col])
            
            r1, r2, rl = compute_rouge_scores(cand, ref)
            _, _, bf = compute_bertscore(cand, ref)
            bl = compute_bleu(cand, ref)
            
            rouge1_list.append(r1)
            rouge2_list.append(r2)
            rougel_list.append(rl)
            bert_list.append(bf)
            bleu_list.append(bl)

        print("\n---------------------------------------------")
        print(" FINAL SYSTEM GENERATION ASSESSMENT MATRIX  ")
        print("---------------------------------------------")
        print(f" Mean ROUGE-1 F1-Score    | {sum(rouge1_list)/len(rouge1_list):.4f}")
        print(f" Mean ROUGE-2 F1-Score    | {sum(rouge2_list)/len(rouge2_list):.4f}")
        print(f" Mean ROUGE-L F1-Score    | {sum(rougel_list)/len(rougel_list):.4f}")
        print(f" Mean BLEU Textual Score  | {sum(bleu_list)/len(bleu_list):.4f}")
        print(f" Mean BERTScore F1-Value  | {sum(bert_list)/len(bert_list):.4f}")
        print("---------------------------------------------\n")
