# Brain Tumor MRI Classification Dataset

This document details the configuration, class specifications, and data partitioning setup for the Brain Tumor Detection MRI dataset.

---

## 🗂️ Class Categories

The dataset consists of T1-weighted, T2-weighted, and FLAIR brain MRI scans classified into 4 distinct categories:

1. **glioma**
   - *Description*: Brain tumors arising from glial cells (astrocytes, oligodendrocytes, or ependymal cells). They are the most common type of primary brain tumor.
   - *Key Features*: Irregular margins, perifocal edema, heterogeneous enhancement.
   
2. **meningioma**
   - *Description*: Typically benign tumors arising from the meninges (membranes covering the brain and spinal cord).
   - *Key Features*: Dural tail sign, well-circumscribed boundaries, homogeneous contrast enhancement.

3. **pituitary**
   - *Description*: Tumors originating in the pituitary gland at the base of the brain (often benign adenomas).
   - *Key Features*: Sellar expansion, compression of the optic chiasm.

4. **notumor**
   - *Description*: Normal, healthy control MRI scans showing no signs of neoplastic growths.
   - *Key Features*: Symmetrical brain hemispheres, normal ventricles, clear sulci and gyri.

---

## 📐 Data Re-organization and Partitioning

The raw dataset files are divided into two main folders provided at the root directory:
- `Brain Tumor Detection from MRI/Training`
- `Brain Tumor Detection from MRI/Testing`

### Preprocessing and Validation Pipeline (`ml/main.py`)
To prevent data leakage and handle class imbalance, the preprocessing pipeline applies the following policy:
1. **Deduplication**: Files with identical content (verified using MD5 hashing) are detected. Only the first unique file is retained, and any duplicate versions (e.g., from vendor augmentations) are relocated to `ml/artifacts/duplicates/` to prevent contamination.
2. **Leakage Control**: The system checks if identical image hashes exist between Training and Testing sets. If any identical image spans across splits, it is flagged and blocked to guarantee no cross-split leakage.
3. **Stratified Split**: The verified clean dataset is split into `Train`, `Validation`, and `Test` subsets using a stratified splitter. This preserves class ratios across all splits:
   - **Train**: 80% (used for backward propagation weight tuning)
   - **Validation**: 10% (used for early stopping trigger and hyperparameter evaluation)
   - **Test**: 10% (used for final unseen model performance evaluation)
