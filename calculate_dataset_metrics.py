import json
import time
from collections import Counter

def main() -> None:
    print("Starting comprehensive dataset metrics calculation...")
    start_time = time.time()
    
    total_neurons = 0
    total_super_tokens = 0
    
    special_counts = Counter()
    geo_counts = Counter()
    inv_counts = Counter()
    reg_counts = Counter()
    
    jump_count = 0
    
    with open("tokenized_dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            total_neurons += 1
            try:
                data = json.loads(line)
                seq = data.get("sequence", [])
                total_super_tokens += len(seq)
                
                for item in seq:
                    if isinstance(item, list) and len(item) == 3:
                        tok1, tok2, tok3 = item
                        if tok1 == tok2 == tok3:
                            if tok1.startswith("<JUMP_"):
                                jump_count += 1
                            else:
                                special_counts[tok1] += 1
                        else:
                            inv_counts[tok1] += 1
                            geo_counts[tok2] += 1
                            reg_counts[tok3] += 1
            except Exception:
                pass
                
            if total_neurons % 1000 == 0:
                print(f"Processed {total_neurons} neurons...")
            if total_neurons >= 5000:
                break
                
    elapsed = time.time() - start_time
    
    # Extrapolate to the known total of 182,329 neurons
    extrapolation_factor = 182329 / total_neurons
    total_super_tokens = int(total_super_tokens * extrapolation_factor)
    for k in special_counts:
        special_counts[k] = int(special_counts[k] * extrapolation_factor)
    jump_count = int(jump_count * extrapolation_factor)
    
    # We don't extrapolate unique vocab counts because 5000 neurons is plenty to hit all 512 clusters
    
    total_tokens = total_super_tokens * 3
    nodes_represented = sum(geo_counts.values()) + total_neurons
    fertility = total_super_tokens / max(1, nodes_represented)
    
    print("\n" + "="*50)
    print("DATASET METRICS")
    print("="*50)
    print(f"Total Neurons: {total_neurons:,}")
    print(f"Total Super-Tokens (Sequence Steps): {total_super_tokens:,}")
    print(f"Total Individual Tokens: {total_tokens:,}")
    print(f"Estimated Physical Nodes: {nodes_represented:,}")
    print(f"Fertility Rate (Super-Tokens per Node): {fertility:.2f} (Expected ~1.0-1.5 due to BIF/POP)")
    
    print("\n--- Vocabulary Usage ---")
    print(f"Unique GEO tokens used: {len(geo_counts)} / 512")
    print(f"Unique INV tokens used: {len(inv_counts)} / 512")
    print(f"Unique REG tokens used: {len(reg_counts)}")
    
    print("\n--- Structural Token Frequencies ---")
    print(f"<BIF> (Bifurcations): {special_counts.get('<BIF>', 0):,}")
    print(f"<END_BIF> (End of Bifurcations): {special_counts.get('<END_BIF>', 0):,}")
    print(f"<POP> (Branch Returns): {special_counts.get('<POP>', 0):,}")
    print(f"<JUMP> (Spatial Gaps): {jump_count:,}")
    print(f"<START> / <END>: {special_counts.get('<START>', 0):,} / {special_counts.get('<END>', 0):,}")
    
    print(f"\nTime taken: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
