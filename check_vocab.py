import json
with open('Neuro_Model/config.json', 'r') as f:
    config = json.load(f)
print("Model Vocab sizes:", config['architecture_parameters']['vocab_sizes'])
