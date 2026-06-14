import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ========== 配置 ==========
INPUT_FILE = "less 0.35.xlsx"
OUTPUT_FILE = "drug_likeness_results.xlsx"
ID_COL = "ID"
SMILES_COL = "Smiles"

# ========== 定义计算函数 ==========
def compute_drug_likeness(smiles):
    """
    输入 SMILES 字符串，返回包含类药性参数的字典；
    若 SMILES 无效，返回含 None 的字典。
    """
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return {
            "MW": None,
            "LogP": None,
            "HBD": None,
            "HBA": None,
            "RotBonds": None,
            "TPSA": None,
            "Lipinski_violations": None,
            "Veber_violations": None,
            "Pass_Lipinski": None,
            "Pass_Veber": None,
            "Valid_SMILES": False
        }

    # 基本描述符
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotb = Lipinski.NumRotatableBonds(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)

    # Lipinski 五规则 (MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10)
    lipinski_violations = 0
    if mw > 500: lipinski_violations += 1
    if logp > 5: lipinski_violations += 1
    if hbd > 5: lipinski_violations += 1
    if hba > 10: lipinski_violations += 1

    # Veber 规则 (可旋转键 <= 10, TPSA <= 140)
    veber_violations = 0
    if rotb > 10: veber_violations += 1
    if tpsa > 140: veber_violations += 1

    return {
        "MW": round(mw, 2),
        "LogP": round(logp, 2),
        "HBD": hbd,
        "HBA": hba,
        "RotBonds": rotb,
        "TPSA": round(tpsa, 2),
        "Lipinski_violations": lipinski_violations,
        "Veber_violations": veber_violations,
        "Pass_Lipinski": (lipinski_violations <= 1),   # 通常允许 1 条违背
        "Pass_Veber": (veber_violations == 0),
        "Valid_SMILES": True
    }

# ========== 主程序 ==========
print(f"正在读取文件：{INPUT_FILE}")
df = pd.read_excel(INPUT_FILE)

# 检查列名
if ID_COL not in df.columns or SMILES_COL not in df.columns:
    raise ValueError(f"输入文件必须包含 '{ID_COL}' 和 '{SMILES_COL}' 列")

print(f"化合物总数：{len(df)}")

# 逐行计算
results = []
for idx, row in df.iterrows():
    smi = row[SMILES_COL]
    desc = compute_drug_likeness(smi)
    desc[ID_COL] = row[ID_COL]
    desc[SMILES_COL] = smi
    results.append(desc)

# 构建结果 DataFrame
result_df = pd.DataFrame(results)

# 列顺序
cols = [
    ID_COL, SMILES_COL, "Valid_SMILES",
    "MW", "LogP", "HBD", "HBA", "RotBonds", "TPSA",
    "Lipinski_violations", "Pass_Lipinski",
    "Veber_violations", "Pass_Veber"
]
result_df = result_df[cols]

# 输出
result_df.to_excel(OUTPUT_FILE, index=False)

print(f"\n计算完成！结果已保存至：{OUTPUT_FILE}")
print(f"通过 Lipinski 规则的化合物数：{result_df['Pass_Lipinski'].sum()}")
print(f"通过 Veber 规则的化合物数：{result_df['Pass_Veber'].sum()}")
print(f"同时通过两个规则的化合物数：{(result_df['Pass_Lipinski'] & result_df['Pass_Veber']).sum()}")
print(f"无效 SMILES 数：{(~result_df['Valid_SMILES']).sum()}")