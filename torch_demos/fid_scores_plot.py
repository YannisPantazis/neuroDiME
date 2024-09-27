import json
import matplotlib.pyplot as plt
import os

# Define the file paths and labels
file_paths = {
    'Renyi-CC-WCR': 'fid_scores_Renyi-CC-WCR.json',
    'alpha-LT': 'fid_scores_alpha-LT.json',
    'chi-squared-LT': 'fid_scores_chi-squared-LT.json',
    'IPM': 'fid_scores_IPM.json',
    'JS-LT': 'fid_scores_JS-LT.json',
    'KLD-DV': 'fid_scores_KLD-DV.json',
    'Renyi-CC': 'fid_scores_Renyi-CC.json',
    'Renyi-DV': 'fid_scores_Renyi-DV.json',
    'rescaled-Renyi-CC': 'fid_scores_rescaled-Renyi-CC.json',
    'squared-Hel-LT': 'fid_scores_squared-Hel-LT.json',
}

plt.figure(figsize=(10, 8))

for label, file_path in file_paths.items():
    with open(file_path, 'r') as f:
        fid_scores = json.load(f)
        plt.plot(fid_scores, label=label)
        
plt.xlabel('Epochs')
plt.ylabel('FID Score')
plt.legend()
plt.title('FID Scores for Different Divergences - MNIST')
plt.savefig('fid_scores_plot.png')