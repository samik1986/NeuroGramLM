import json
import time

def main():
    print("Starting token count...")
    start_time = time.time()
    total_lines = 0
    total_sequence_items = 0
    
    with open("tokenized_dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            try:
                data = json.loads(line)
                seq = data.get("sequence", [])
                total_sequence_items += len(seq)
            except Exception:
                pass
                
            if total_lines % 100000 == 0:
                print(f"Processed {total_lines} lines...")
                
    elapsed = time.time() - start_time
    print(f"\n--- Statistics ---")
    print(f"Total Neurons (lines): {total_lines}")
    print(f"Total Sequence Items (super-tokens): {total_sequence_items}")
    print(f"Total Individual Tokens: {total_sequence_items * 3}")
    print(f"Time taken: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
