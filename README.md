# Dual-Attention FERNet for Facial Expression Recognition with Data Augmentation
This project uses the FER2013 dataset for facial expression recognition. 
Due to class imbalance, we applied data augmentation (flipping, rotation, contrast inversion) to balance the dataset.

DualAttention_FERNet
A dual-branch CNN with 3×3 and 5×5 convolutions, combined with an AttentionLayer to enhance feature selection. 
The model effectively focuses on emotion-relevant regions, improving recognition accuracy.

Training Details:
Optimizer: Adam
Loss: Categorical Cross-Entropy
Epochs: 25 | Batch Size: 32
Total Parameters: 5.1M

Results:
Accuracy: 73.49%
Precision: 73.47% | Recall: 73.49% | F1-Score: 73.42%
The attention mechanism helps the model focus on key facial features, enhancing performance.
