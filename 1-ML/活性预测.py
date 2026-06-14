#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
from warnings import filterwarnings
import time
import pandas as pd
import numpy as np
from sklearn import svm, metrics, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import auc, accuracy_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect


# In[2]:


df = pd.read_csv('prediction.csv', sep=',')
print("Dataframe shape:", df.shape)
print(df.head())


# In[3]:


from rdkit.Chem.MolStandardize import rdMolStandardize
md = rdMolStandardize.MetalDisconnector()
lfc = rdMolStandardize.LargestFragmentChooser()
uncharger = rdMolStandardize.Uncharger()

def canonical_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)

        # 1) 断开金属（Na/K/Ca...）
        mol = md.Disconnect(mol)

        # 2) 取母体主片段（去掉游离的小碎片/对离子）
        mol = lfc.choose(mol)

        # 3) 去电荷（统一为中性母体，活性预测通常更合适）
        mol = uncharger.uncharge(mol)

        # 4) 输出规范化 SMILES
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None

from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
morgan2_gen = GetMorganGenerator(radius=2, fpSize=2048)

def smiles_to_fp(smiles, method="morgan2", n_bits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if method == "maccs":
            return np.array(MACCSkeys.GenMACCSKeys(mol))
        elif method == "morgan2":
            return np.array(morgan2_gen.GetFingerprint(mol))
        else:
            return np.array(morgan2_gen.GetFingerprint(mol))
    except Exception:
        return None


# In[7]:


df["canonical_smiles"] = df["Smiles"].apply(canonical_smiles)
df.head()


# In[9]:


df_cleaned = df.drop_duplicates(
    subset=["ID", "canonical_smiles"]
).reset_index(drop=True)
df_cleaned.head()


# In[11]:


print("原始:", df.shape)
print("去重后:", df_cleaned.shape)


# ## 添加指纹

# In[12]:


df_fp = df_cleaned.copy()

df_fp["fp"] = df_fp["canonical_smiles"].apply(
    lambda s: smiles_to_fp(s, method="morgan2", n_bits=2048)
)


# In[13]:


df_fp = df_fp.dropna(subset=["fp"]).reset_index(drop=True)


# In[14]:


print("原始:", df_cleaned.shape)
print("生成 fp 后:", df_fp.shape)

df_fp.head()


# ## 活性预测

# In[15]:


import os
import joblib

model_paths = {
    "svm":    "svm_model.pkl"
}

models = {name: joblib.load(path) for name, path in model_paths.items()}


# In[ ]:


import numpy as np

# 确保和训练时格式一致
X = np.array(df_fp["fp"].tolist())

df_pred = df_fp.copy()

for model_name, model in models.items():
    # 二分类预测
    df_pred[f"{model_name}_pred"] = model.predict(X)
    
    # 概率预测（如果模型支持）
    if hasattr(model, "predict_proba"):
        df_pred[f"{model_name}_prob"] = model.predict_proba(X)[:, 1]
    else:
        df_pred[f"{model_name}_prob"] = np.nan

# 查看结果
df_pred[[
    "svm_pred", "svm_prob"
]].head()


# In[ ]:


df_pred.to_csv(
    "df_predicted.csv",
    index=False,
    encoding="utf-8-sig"
)

print("✅ df_predicted 已保存为 df_predicted.csv")

