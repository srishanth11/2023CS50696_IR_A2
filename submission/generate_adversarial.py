import json
from collections import defaultdict
from submission.corpus_utils import load_corpus
from submission.lm_utils import CollectionStats

def generate_adversarial_set(corpus_path: str, queries_path: str, output_path: str):
    print("Corpus load hot ahe...")
    corpus = load_corpus(corpus_path)
    stats = CollectionStats.from_corpus(corpus)
    
    # Highest document length / high noise aslele docs shodha
    sorted_docs = sorted(
        stats.doc_lengths.items(), 
        key=lambda item: item[1], 
        reverse=True
    )
    
    # Top noisy/off-topic docs chi list banwa
    noisy_doc_ids = [doc_id for doc_id, length in sorted_docs[:50]]
    
    adversarial_map = {}
    
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            qid = parts[0]
            # Pratyek query sathi 3 to 5 adversarial docs assign kara
            adversarial_map[qid] = noisy_doc_ids[:5]
            
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(adversarial_map, f, indent=2)
        
    print(f"Adversarial set successfully create jhala: {output_path}")

if __name__ == "__main__":
    generate_adversarial_set(
        corpus_path="data/full/corpus.jsonl",
        queries_path="data/full/queries_dev.tsv",
        output_path="submission/adversarial_set.json"
    )