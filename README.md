Integrated Computational Framework for Molecular Modeling: Machine Learning Prediction, Structural Similarity Analysis, and Drug-Likeness Evaluation
Abstract
This repository presents an integrated computational framework for small-molecule analysis in drug discovery. The framework consists of three independent but complementary modules: (a) a machine learning-based activity prediction model using a Support Vector Machine (SVM), (b) a molecular similarity assessment module based on Morgan fingerprints and Tanimoto similarity, and (c) a rule-based drug-likeness evaluation module. The system is designed for reproducible virtual screening and is intended for use by researchers with basic Python proficiency in chemoinformatics.

1. Overview of the Framework
1.1 Machine Learning-Based Activity Prediction
A supervised learning model based on Support Vector Machine (SVM) is employed for binary or continuous activity prediction. A pretrained model is provided to enable direct inference without retraining.
- Algorithm: Support Vector Machine (SVM)
- Implementation: scikit-learn
- Model file: svm_model.pkl
1.2 Molecular Similarity Analysis
Structural similarity between molecules is quantified using Morgan fingerprints and the Tanimoto coefficient. This module enables nearest-neighbor retrieval in chemical space.
- Representation: Morgan fingerprints
- Similarity metric: Tanimoto coefficient
- Purpose: virtual screening and compound prioritization
1.3 Drug-Likeness Evaluation
A rule-based filtering strategy is implemented to assess the drug-likeness of compounds based on physicochemical property thresholds.
- Method: Rule-based filtering
- Output: filtered compound subset

2. Repository Structure
1-ML/ML_26.py;活性预测.py; svm_model.pkl; prediction.csv
2-similarity/morgan_tanimoto_top5.py; reference compounds.xlsx; test compounds.xlsx
3-drug likeness/drug_likeness.py; less 0.35.xlsx

3. Requirements
3.1 Software Environment
Python ≥ 3.8
3.2 Dependencies
The required Python packages are listed below:
pip install numpy pandas scikit-learn rdkit openpyxl

4. Usage Instructions
Each module can be executed independently.
4.1 Activity Prediction (SVM Model)
cd 1-ML  python 活性预测.py
Input: prediction.csv
Note:The pretrained model (svm_model.pkl) is directly used for inference.
4.2 Molecular Similarity Analysis
cd 2-similarity
python morgan_tanimoto_top5.py
Input: test compounds.xlsx; reference compounds.xlsx
Output: Top-5 structurally similar compounds per query molecule
4.3 Drug-Likeness Evaluation
cd 3-drug likeness  python drug_likeness.py
Input: Filtered compound dataset (less 0.35.xlsx)

5. Input Data Format
Input molecules should be provided in SMILES format or structured tabular files (Excel format) containing compound identifiers and molecular representations.

6. Reproducibility Statement
All computational models, scripts, and example datasets required to reproduce the results are included in this repository. The pretrained model is provided to ensure full reproducibility without retraining.

7. System Requirements
Python 3.8 or higher
RDKit for cheminformatics operations
CPU-based execution supported (no GPU required for inference)

8. Limitations
The performance of the machine learning model is dependent on the diversity and quality of the training dataset. The framework is intended for virtual screening and compound prioritization rather than experimental validation or clinical decision-making.

