# Working Space for Cross Attention Fusion Implementation

| SN | Date       | Epochs | ACC    | AUC-ROC | Remarks          |
|----|------------|--------|--------|---------|------------------|
| 1  | 2 April    | 10     | 0.5644 | 0.7259  | self-attn<br>local cpu 48mins |
| 2  | 2 April    | 20     | 0.5743 | 0.7622  | self-attn<br>colab t4 gpu 12mins |
| 3  | 2 April    | 20     | 0.8911 | 0.9499  | self-attn<br>switched text from notes to findings |
| 4  | 2 April    | 20     | 0.8614 | 0.9127  | cross-attn<br>align train-test split with marcus<br>overfitting |