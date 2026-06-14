#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import joblib


# ## 1.活性数据处理

# In[ ]:


df = pd.read_csv('C:\\Users\\31584\\Desktop\\wangliqing_project\\ML_compounds\\Training_database.csv', sep=',')
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


# In[4]:


# 1) 生成 canonical_smiles 和 fp
compound_df = df.copy()
compound_df["canonical_smiles"] = compound_df["smiles"].apply(canonical_smiles)
compound_df["fp"] = compound_df["canonical_smiles"].apply(
    lambda s: smiles_to_fp(s, method="morgan2", n_bits=2048)
)

# 2) 后处理：去掉失败的 + 去重
compound_df = compound_df.dropna(subset=["canonical_smiles", "fp"]).reset_index(drop=True)
compound_df = compound_df.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)

# 3) 保存：包含RDKit规范化后的SMILES
# 如果你不想把fp保存进csv（因为是数组会很大），就先drop掉
save_df = compound_df.drop(columns=["fp"])
save_df.to_csv("3CL_cleaned.csv", index=False, encoding="utf-8-sig")

print("已保存: 3CL_cleaned.csv")


# In[5]:


compound_df.head()


# In[6]:


fp0 = compound_df["fp"].iloc[0]
print(type(fp0))
lengths = compound_df["fp"].apply(len)
print(lengths.value_counts().head())


# ### 普通采样办法

# In[9]:


from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

x = compound_df.fp.tolist()
y = compound_df.Label.tolist()

# 转换为适合划分的格式
X = np.array(x)  
y = np.array(y)

# 打印原始数据分布
print("原始数据分布:", Counter(y))
print("特征矩阵形状:", X.shape)

# 普通划分训练集和测试集（使用分层采样保持类别比例）
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=12345, stratify=y)

# 打印划分后的分布
print("\n划分后数据分布:")
print("训练集大小:", len(train_x))
print("测试集大小:", len(test_x))
print("训练集类别分布:", Counter(train_y))
print("测试集类别分布:", Counter(test_y))


# 如果你想保留所有拆分结果
splits = [train_x, test_x, train_y, test_y]

# 现在可以用于模型训练
# 训练集: train_x, train_y
# 测试集: test_x, test_y


# ### 权重采样办法

# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import numpy as np

# 原始数据
X = np.array(compound_df.fp.tolist())   # (2149, 2048)
y = np.array(compound_df.label.tolist())

print("原始数据集大小:", X.shape[0])
print("原始数据分布:", Counter(y))
print("特征矩阵形状:", X.shape)
print("-" * 50)

# 划分训练集 / 测试集
train_x, test_x, train_y, test_y = train_test_split(
    X, y,
    test_size=0.2,
    random_state=12345,
    stratify=y
)

# 打印训练 / 测试集信息
print("训练集大小:", train_x.shape[0])
print("训练集特征形状:", train_x.shape)
print("训练集标签分布:", Counter(train_y))
print("-" * 30)

print("测试集大小:", test_x.shape[0])
print("测试集特征形状:", test_x.shape)
print("测试集标签分布:", Counter(test_y))
print("-" * 50)

# 计算类别权重（仅基于训练集）
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_y),
    y=train_y
)

class_weight_dict = dict(zip(np.unique(train_y), class_weights))

print(
    f"类别权重 - 负类(0): {class_weight_dict[0]:.2f}, "
    f"正类(1): {class_weight_dict[1]:.2f}"
)
print("正样本将在训练中获得更高权重")


# In[8]:


from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score, 
                            f1_score, confusion_matrix, cohen_kappa_score,
                            precision_score, precision_recall_curve, 
                            average_precision_score)
import matplotlib.pyplot as plt
import numpy as np

def model_performance(ml_model, test_x, test_y, verbose=True):
    # score：优先概率，否则用 decision_function
    if hasattr(ml_model, "predict_proba"):
        test_score = ml_model.predict_proba(test_x)[:, 1]
        test_pred = ml_model.predict(test_x)
    else:
        test_score = ml_model.decision_function(test_x)
        test_pred = (test_score >= 0).astype(int)  # SVM decision_function 0 为默认阈值

    accuracy = accuracy_score(test_y, test_pred)
    sens = recall_score(test_y, test_pred)
    prec = precision_score(test_y, test_pred, zero_division=0)
    spec = recall_score(test_y, test_pred, pos_label=0)
    auc = roc_auc_score(test_y, test_score)
    f1 = f1_score(test_y, test_pred, zero_division=0)
    cm = confusion_matrix(test_y, test_pred)
    kappa = cohen_kappa_score(test_y, test_pred)

    precision_vals, recall_vals, _ = precision_recall_curve(test_y, test_score)
    ap_score = average_precision_score(test_y, test_score)

    if verbose:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Sensitivity/Recall: {sens:.4f}")
        print(f"Specificity: {spec:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Precision: {prec:.4f}")
        print(f"Average Precision (AP): {ap_score:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")

    return accuracy, sens, spec, auc, f1, cm, kappa, prec, precision_vals, recall_vals, ap_score


# In[9]:


import matplotlib as mpl
mpl.rcParams["font.family"] = "Times New Roman"


# ## 1.1 ROC曲线绘制代码

# In[10]:


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def plot_roc_curves_for_models(models, test_x, test_y, save_png=False):
    fig, ax = plt.subplots(figsize=(8, 6))  # 关键：加宽

    for model in models:
        ml_model = model["model"]
        test_prob = ml_model.predict_proba(test_x)[:, 1]
        fpr, tpr, _ = roc_curve(test_y, test_prob)
        auc = roc_auc_score(test_y, test_prob)

        label = f"{model['label'].replace('Model_', '')} (AUC={auc:.4f})"
        ax.plot(fpr, tpr, label=label)

    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")

    # 关键：给右侧留空间（0.72~0.8 之间自己微调）
    fig.subplots_adjust(right=0.72)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    if save_png:
        fig.savefig("roc_auc_curve.png", dpi=1200, bbox_inches="tight", transparent=True)

    return fig


# ## 1.2 PR曲线代码

# In[11]:


from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def plot_pr_curves_for_models(models, test_x, test_y,
                              figsize=(8, 6),
                              baseline=True,
                              legend_outside=True,
                              save_png=False,
                              save_path="pr_curve.png"):
    """
    models: list of dicts, e.g. [{"label":"Model_RF","model":clf}, ...]
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Baseline: precision = positive class prevalence
    if baseline:
        pos_rate = (test_y == 1).mean()
        ax.hlines(pos_rate, 0, 1, linestyles="--", label=f"Baseline (pos={pos_rate:.3f})")

    for item in models:
        m = item["model"]
        name = item.get("label", "model").replace("Model_", "")

        # score: prefer predict_proba, else decision_function
        if hasattr(m, "predict_proba"):
            y_score = m.predict_proba(test_x)[:, 1]
        elif hasattr(m, "decision_function"):
            y_score = m.decision_function(test_x)
        else:
            raise ValueError(f"{name} has neither predict_proba nor decision_function")

        precision, recall, _ = precision_recall_curve(test_y, y_score)
        ap = average_precision_score(test_y, y_score)

        ax.plot(recall, precision, lw=2, label=f"{name} (AP={ap:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    if legend_outside: 
        fig.subplots_adjust(right=0.72)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    else:
        ax.legend(loc="best", frameon=False)

    if save_png:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight", transparent=True)

    return fig


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix", cmap="Blues", normalize=False):
    """
    绘制混淆矩阵（支持归一化）
    
    Parameters
    ----------
    y_true : array-like
        实际标签
    y_pred : array-like
        预测标签
    labels : list, optional
        标签类别顺序（如 [0, 1]）
    title : str
        图标题
    cmap : str
        热图颜色方案
    normalize : bool
        是否归一化显示比例（百分比）
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
    else:
        fmt = "d"

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                square=True, cbar=False, linewidths=0.5, linecolor='gray')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


# ## 2.1 随机森林模型

# In[38]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, confusion_matrix


# 初始化随机森林模型
rf_model = RandomForestClassifier(random_state=1234,class_weight=class_weight_dict)

# 设置超参数搜索空间
rf_param_grid = {
    # 树的数量和复杂度
   'n_estimators': range(10,500,10),
    'max_depth': range(10,50,5),
}

# 网格搜索 + 5折交叉验证
rf_grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=rf_param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='roc_auc'
)

rf_grid_search.fit(train_x, train_y)

# 打印最佳参数
print("Best parameters for Random Forest:", rf_grid_search.best_params_)

# 获取最优模型
best_rf_model = rf_grid_search.best_estimator_

# 用最优模型在测试集上进行预测和评估
accuracy, sens, spec, auc, f1, cm, kappa, prec, precision_vals, recall_vals, ap_score = \
    model_performance(best_rf_model, test_x, test_y, verbose=True)
models = []
# 例如 RF 训练完后
models.append({"label": "Model_RF", "model": best_rf_model})


# In[30]:


import joblib
joblib.dump(best_rf_model, 'random_forest_model.pkl')


# ## 2.2 SVM

# In[14]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

svm_model = SVC(kernel="rbf", probability=True, random_state=42, class_weight=class_weight_dict)

svm_param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
}

svm_grid_search = GridSearchCV(
    estimator=svm_model,
    param_grid=svm_param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring="roc_auc"
)

svm_grid_search.fit(train_x, train_y)
best_svm_model = svm_grid_search.best_estimator_

print("Best parameters for SVM:", svm_grid_search.best_params_)

accuracy, sens, spec, auc, f1, cm, kappa, prec, precision_vals, recall_vals, ap_score = \
    model_performance(best_svm_model, test_x, test_y, verbose=True)

models.append({"label": "Model_SVM", "model": best_svm_model})


# In[29]:


import joblib
joblib.dump(best_svm_model, 'random_svm_model.pkl')


# ## 2.3 逻辑回归

# In[15]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

logreg_model = LogisticRegression(
    random_state=42,
    max_iter=5000,
    class_weight=class_weight_dict,
    solver="saga"
)

logreg_param_grid = {
    "penalty": ["l1", "l2"],
    "C": [0.01, 0.1, 1, 10],
}

logreg_grid_search = GridSearchCV(
    estimator=logreg_model,
    param_grid=logreg_param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring="roc_auc"
)

logreg_grid_search.fit(train_x, train_y)
best_logreg_model = logreg_grid_search.best_estimator_

print("Best parameters for Logistic Regression:", logreg_grid_search.best_params_)

accuracy, sens, spec, auc, f1, cm, kappa, prec, precision_vals, recall_vals, ap_score = \
    model_performance(best_logreg_model, test_x, test_y, verbose=True)

models.append({"label": "Model_logreg", "model": best_logreg_model})


# ## 2.4 MLP

# In[22]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

mlp_pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("mlp", MLPClassifier(
        random_state=42,
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=20
    ))
])

mlp_param_dist = {
    "mlp__hidden_layer_sizes": [(50,), (100,), (200,), (100, 50), (200, 100)],
    "mlp__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
    "mlp__learning_rate_init": [1e-4, 1e-3, 1e-2],
    "mlp__batch_size": [32, 64, 128]
}

mlp_search = RandomizedSearchCV(
    estimator=mlp_pipe,
    param_distributions=mlp_param_dist,
    n_iter=30,           # 比全网格快很多
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring="roc_auc",
    random_state=42
)

mlp_sample_weight = np.array(
    [class_weight_dict[y_i] for y_i in train_y]
)

mlp_search.fit(train_x, train_y,
    mlp__sample_weight=mlp_sample_weight)
best_mlp_model = mlp_search.best_estimator_
print("Best parameters for Neural Network:", mlp_search.best_params_)

accuracy, sens, spec, auc, f1, cm, kappa, prec, precision_vals, recall_vals, ap_score = \
    model_performance(best_mlp_model, test_x, test_y, verbose=True)

models.append({"label": "Model_MLP", "model": best_mlp_model})


# ## 2.5 K-最近邻 KNN

# In[19]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 确保是0/1或bool
train_x_knn = (train_x > 0).astype(np.uint8)
test_x_knn  = (test_x > 0).astype(np.uint8)

knn_model = KNeighborsClassifier()

knn_param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
    "weights": ["uniform", "distance"],
    "metric": ["jaccard", "hamming"]   # 优先 jaccard
}

knn_grid_search = GridSearchCV(
    estimator=knn_model,
    param_grid=knn_param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring="roc_auc"
)

knn_grid_search.fit(train_x_knn, train_y)
print("Best parameters for KNN:", knn_grid_search.best_params_)

best_knn_model = knn_grid_search.best_estimator_

accuracy, sens, spec, auc, f1, cm, kappa, prec, precision_vals, recall_vals, ap_score = \
    model_performance(best_knn_model, test_x_knn, test_y, verbose=True)

models.append({"label": "Model_KNN", "model": best_knn_model})


# In[21]:


import joblib
joblib.dump(best_knn_model, 'random_knn_model.pkl')


# ## 2.6 Adaboost

# In[23]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# ================= AdaBoost 模型 =================
ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    random_state=42
)

ada_param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.1, 0.5, 1.0],
    "estimator__max_depth": [1, 2, 3]
}

ada_grid_search = GridSearchCV(
    estimator=ada_model,
    param_grid=ada_param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring="roc_auc"
)

# ====== ✅ 关键：class_weight_dict → sample_weight ======
ada_sample_weight = np.array(
    [class_weight_dict[int(label)] for label in train_y]
)

# ====== ✅ 关键：在 fit 时传入 sample_weight ======
ada_grid_search.fit(
    train_x,
    train_y,
    sample_weight=ada_sample_weight
)

print("Best parameters for AdaBoost:", ada_grid_search.best_params_)

best_ada_model = ada_grid_search.best_estimator_

accuracy, sens, spec, auc, f1, cm, kappa, prec, precision_vals, recall_vals, ap_score = \
    model_performance(best_ada_model, test_x, test_y, verbose=True)

models.append({"label": "Model_AdaBoost", "model": best_ada_model})


# ## 2.7 XGBoost

# In[24]:


from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# ====== ✅ 关键：由 class_weight_dict 推导 ======
scale_pos_weight = class_weight_dict[1] / class_weight_dict[0]

xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=42,
    n_jobs=1,
    scale_pos_weight=scale_pos_weight
)

xgb_param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [200, 400, 800],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.5, 0.8, 1.0],
    "gamma": [0, 1]
}

xgb_grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=xgb_param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring="roc_auc"
)

xgb_grid_search.fit(train_x, train_y)

print("Best parameters for XGBoost:", xgb_grid_search.best_params_)

best_xgb_model = xgb_grid_search.best_estimator_

accuracy, sens, spec, auc, f1, cm, kappa, prec, precision_vals, recall_vals, ap_score = \
    model_performance(best_xgb_model, test_x, test_y, verbose=True)

models.append({"label": "Model_XGBoost", "model": best_xgb_model})


# In[23]:


import joblib
joblib.dump(best_xgb_model, 'random_xgb_model.pkl')


# ## 2.8 CatBoost

# In[25]:


from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

# ====== ✅ 关键：由 class_weight_dict 构造 ======
cat_class_weights = [class_weight_dict[0], class_weight_dict[1]]

cat_model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    random_state=42,
    verbose=0,
    thread_count=1,
    class_weights=cat_class_weights
)

cat_param_grid = {
    "depth": [3, 5, 7],
    "learning_rate": [0.03, 0.1],
    "iterations": [200, 300],
    "l2_leaf_reg": [1, 3]
}

cat_grid_search = GridSearchCV(
    estimator=cat_model,
    param_grid=cat_param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring="roc_auc"
)

cat_grid_search.fit(train_x, train_y)

print("Best parameters for CatBoost:", cat_grid_search.best_params_)

best_cat_model = cat_grid_search.best_estimator_

accuracy, sens, spec, auc, f1, cm, kappa, prec, precision_vals, recall_vals, ap_score = \
    model_performance(best_cat_model, test_x, test_y, verbose=True)

models.append({"label": "Model_CatBoost", "model": best_cat_model})


# ## 2.9 LightGBM

# In[26]:


from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

lgb_model = LGBMClassifier(
    random_state=42,
    verbose=-1,
    n_jobs=1,
    class_weight=class_weight_dict
)

lgb_param_grid = {
    "max_depth": [5, -1],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [100, 200],
    "num_leaves": [31, 63],
    "min_child_samples": [20, 50]
}

lgb_grid_search = GridSearchCV(
    estimator=lgb_model,
    param_grid=lgb_param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring="roc_auc"
)

lgb_grid_search.fit(train_x, train_y)

print("Best parameters for LightGBM:", lgb_grid_search.best_params_)

best_lgb_model = lgb_grid_search.best_estimator_

accuracy, sens, spec, auc, f1, cm, kappa, prec, precision_vals, recall_vals, ap_score = \
    model_performance(best_lgb_model, test_x, test_y, verbose=True)

models.append({"label": "Model_LightGBM", "model": best_lgb_model})


# In[26]:


import joblib
joblib.dump(best_lgb_model, 'random_lgb_model.pkl')


# ## 2.10 朴素贝叶斯

# In[21]:


from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB  # 适用于二进制特征

# 初始化伯努利朴素贝叶斯模型（适用于二进制特征）
nb_model = BernoulliNB()

# 设置超参数搜索空间
nb_param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],  # 平滑参数
}

# 网格搜索 + 5折交叉验证
nb_grid_search = GridSearchCV(
    estimator=nb_model,
    param_grid=nb_param_grid,
    cv=5,                  # 5折交叉验证
    n_jobs=-1,             # 使用所有CPU核心
    verbose=2,             # 打印详细信息
    scoring='roc_auc'     # 评估指标
)

# 用训练集进行超参数调优
nb_grid_search.fit(train_x, train_y)

# 打印最佳参数
print("\nBest parameters for Naive Bayes:", nb_grid_search.best_params_)

# 获取最优模型
best_nb_model = nb_grid_search.best_estimator_

accuracy, sens, spec, auc, f1, cm, kappa, prec, precision_vals, recall_vals, ap_score = \
    model_performance(best_nb_model, test_x, test_y, verbose=True)

models.append({"label": "Model_NB", "model": best_nb_model})


# ## ROC曲线绘制

# In[27]:


models = []

# 1) Random Forest
models.append({"label": "Model_RandomForest", "model": best_rf_model})

# 2) SVM (RBF)
models.append({"label": "Model_SVM_RBF", "model": best_svm_model})

# 3) Logistic Regression
models.append({"label": "Model_LogisticRegression", "model": best_logreg_model})

# 4) MLP (Neural Network)
models.append({"label": "Model_MLP", "model": best_mlp_model})

# 5) k-Nearest Neighbors
models.append({"label": "Model_KNN", "model": best_knn_model})

# 6) AdaBoost
models.append({"label": "Model_AdaBoost", "model": best_ada_model})

# 7) XGBoost
models.append({"label": "Model_XGBoost", "model": best_xgb_model})

# 8) CatBoost
models.append({"label": "Model_CatBoost", "model": best_cat_model})

# 9) LightGBM
models.append({"label": "Model_LightGBM", "model": best_lgb_model})

# 10) Bernoulli Naive Bayes
models.append({"label": "Model_BernoulliNB", "model": best_nb_model})


# In[35]:


import pandas as pd

results = []

for item in models:
    label = item["label"]
    model = item["model"]

    accuracy, sens, spec, auc, f1, cm, kappa, prec, precision_vals, recall_vals, ap_score = \
        model_performance(model, test_x, test_y, verbose=False)

    results.append({
        "model": label,
        "accuracy": accuracy,
        "recall": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "auc": auc,
        "ap": ap_score,
        "kappa": kappa,
        "tp": cm[1, 1],
        "fp": cm[0, 1],
        "tn": cm[0, 0],
        "fn": cm[1, 0],
    })

results_df = pd.DataFrame(results)
results_df


# In[39]:


results_df.to_csv(
    "C:\\Users\\31584\\Desktop\\wangliqing_project\\ML_compounds\\模型评估指标.csv",
    index=False,
    encoding="utf-8-sig"
)


# In[28]:


plot_roc_curves_for_models(models, test_x, test_y, save_png=True)
plt.show()


# ## PR曲线绘制

# In[31]:


fig = plot_pr_curves_for_models(models, test_x, test_y, baseline=True, save_png=True, save_path="pr_curve.png")
plt.show()


# In[32]:


import numpy as np

def get_model_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(X)
        # 归一化到 [0,1] 便于校准/Lift 展示（仅用于展示，不代表真实概率）
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        return s
    else:
        raise ValueError("Model has neither predict_proba nor decision_function")


# In[33]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrices(models, X, y, threshold=0.5, ncols=3, figsize=(12, 8),
                            cmap="Blues", save_png=False, save_path="confusion_matrices.png"):
    """
    models: list of dicts, e.g. [{"label":"Model_RF","model":clf}, ...]
    X, y: test set
    threshold: classification threshold applied to score/prob
    """

    n = len(models)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, item in enumerate(models):
        ax = axes[i]
        name = item["label"].replace("Model_", "")
        model = item["model"]

        # score: prefer predict_proba, else decision_function
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            score = model.decision_function(X)
        else:
            raise ValueError(f"{name} has neither predict_proba nor decision_function")

        pred = (score >= threshold).astype(int)
        cm = confusion_matrix(y, pred)

        # plot matrix
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())

        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        # add white grid lines for clarity
        ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

        # write values with adaptive text color
        thresh = cm.max() * 0.6
        for (r, c), val in np.ndenumerate(cm):
            ax.text(
                c, r, f"{val}",
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=12,
                fontweight="bold"
            )

    # turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Confusion Matrices (threshold={threshold})", y=1.02)
    fig.subplots_adjust(wspace=0.6, hspace=0.8, top=0.88)

    if save_png:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight", transparent=True)

    return fig




# In[34]:


# ===== 使用示例 =====
fig = plot_confusion_matrices(models, test_x, test_y, threshold=0.5, ncols=5, figsize=(18, 7),
                              cmap="Blues", save_png=True, save_path="confusion_matrices.png")
plt.show()


# In[70]:


from sklearn.calibration import calibration_curve

def plot_calibration_curves(models, X, y, n_bins=10, figsize=(7, 6), legend_outside=True):
    fig, ax = plt.subplots(figsize=figsize)

    # 完美校准线
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    for item in models:
        name = item["label"].replace("Model_", "")
        score = get_model_score(item["model"], X)

        frac_pos, mean_pred = calibration_curve(y, score, n_bins=n_bins, strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", lw=2, label=name)

    ax.set_xlabel("Mean predicted value")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")

    if legend_outside:
        fig.subplots_adjust(right=0.72)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    else:
        ax.legend(frameon=False)

    fig.tight_layout()
    return fig


# In[71]:


plot_calibration_curves(models, test_x, test_y, n_bins=10)
plt.show()


# In[72]:


def plot_cumulative_gains(models, X, y, figsize=(7, 6), legend_outside=True):
    fig, ax = plt.subplots(figsize=figsize)
    y = np.asarray(y)

    # 基线：随机选择，命中比例 = 选中比例
    ax.plot([0, 1], [0, 1], "k--", label="Random")

    for item in models:
        name = item["label"].replace("Model_", "")
        score = get_model_score(item["model"], X)

        order = np.argsort(-score)        # 从高到低
        y_sorted = y[order]

        cum_pos = np.cumsum(y_sorted)
        total_pos = cum_pos[-1] if cum_pos.size > 0 else 1

        frac_samples = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
        frac_pos_captured = cum_pos / (total_pos + 1e-12)

        ax.plot(frac_samples, frac_pos_captured, lw=2, label=name)

    ax.set_xlabel("Fraction of samples selected (top-ranked)")
    ax.set_ylabel("Fraction of positives captured")
    ax.set_title("Cumulative Gains Curve")

    if legend_outside:
        fig.subplots_adjust(right=0.72)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    else:
        ax.legend(frameon=False)

    fig.tight_layout()
    return fig


# In[73]:


plot_cumulative_gains(models, test_x, test_y)
plt.show()

