import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity


# ==========================
# 基础函数
# ==========================

def smiles_to_mol(smiles):
    """
    将 SMILES 转换为 RDKit Mol 对象
    """
    if pd.isna(smiles):
        return None

    smiles = str(smiles).strip()

    if smiles == "":
        return None

    mol = Chem.MolFromSmiles(smiles)

    return mol


def get_morgan_fp(smiles, radius=2, n_bits=2048):
    """
    计算 Morgan 指纹

    radius=2 表示 ECFP4
    n_bits=2048 是常用长度，比 1024 更不容易发生位碰撞
    """
    mol = smiles_to_mol(smiles)

    if mol is None:
        return None

    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=radius,
            nBits=n_bits
        )
        return fp

    except Exception:
        return None


def prepare_reference_fps(ref_df, smiles_col, radius=2, n_bits=2048):
    """
    预先计算参考化合物的 Morgan 指纹
    """
    ref_data = []

    for idx, row in ref_df.iterrows():
        smiles = row[smiles_col]
        fp = get_morgan_fp(smiles, radius=radius, n_bits=n_bits)

        if fp is not None:
            ref_data.append({
                "ref_index": idx,
                "ref_smiles": smiles,
                "ref_fp": fp
            })

    return ref_data


def calculate_top5_morgan_tanimoto(query_smiles, ref_data, top_n=5, radius=2, n_bits=2048):
    """
    对单个待测化合物计算 Morgan-Tanimoto 相似度

    返回：
    - top5_mean_similarity
    - max_similarity
    - top1 到 top5 的参考化合物及相似度
    """

    query_fp = get_morgan_fp(query_smiles, radius=radius, n_bits=n_bits)

    if query_fp is None:
        result = {
            "test_smiles": query_smiles,
            "top5_mean_similarity": None,
            "max_similarity": None,
            "error": "待测化合物 SMILES 无效"
        }

        for i in range(top_n):
            result[f"top{i+1}_ref_smiles"] = None
            result[f"top{i+1}_similarity"] = None

        return result

    if len(ref_data) == 0:
        result = {
            "test_smiles": query_smiles,
            "top5_mean_similarity": None,
            "max_similarity": None,
            "error": "参考化合物中没有有效 SMILES"
        }

        for i in range(top_n):
            result[f"top{i+1}_ref_smiles"] = None
            result[f"top{i+1}_similarity"] = None

        return result

    ref_fps = [item["ref_fp"] for item in ref_data]

    # 计算待测化合物与所有参考化合物的 Tanimoto 相似度
    similarities = BulkTanimotoSimilarity(query_fp, ref_fps)

    sim_records = []

    for item, sim in zip(ref_data, similarities):
        sim_records.append({
            "ref_index": item["ref_index"],
            "ref_smiles": item["ref_smiles"],
            "similarity": float(sim)
        })

    # 按相似度从高到低排序
    sim_records = sorted(
        sim_records,
        key=lambda x: x["similarity"],
        reverse=True
    )

    # 取最相似的前 top_n 个参考化合物
    top_records = sim_records[:top_n]

    top_mean_similarity = np.mean([x["similarity"] for x in top_records])
    max_similarity = top_records[0]["similarity"]

    result = {
        "test_smiles": query_smiles,
        "top5_mean_similarity": round(float(top_mean_similarity), 4),
        "max_similarity": round(float(max_similarity), 4),
        "error": None
    }

    for i in range(top_n):
        if i < len(top_records):
            result[f"top{i+1}_ref_smiles"] = top_records[i]["ref_smiles"]
            result[f"top{i+1}_similarity"] = round(float(top_records[i]["similarity"]), 4)
        else:
            result[f"top{i+1}_ref_smiles"] = None
            result[f"top{i+1}_similarity"] = None

    return result


# ==========================
# 主程序
# ==========================

if __name__ == "__main__":

    # 输入文件
    ref_file = "reference compounds.xlsx"
    test_file = "test compounds.xlsx"

    # 列名
    smiles_col = "Smiles"          # SMILES 列名
    id_col = "ID"                  # 待测化合物的 ID 列名（根据实际文件修改）

    # 输出文件
    output_file = "morgan_tanimoto_top5_results.xlsx"

    # Morgan 指纹参数
    RADIUS = 2       # radius=2 即 ECFP4
    N_BITS = 2048    # 2048 更推荐

    # 取最相似的前几个化合物
    TOP_N = 5

    print("正在读取参考化合物文件...")
    ref_df = pd.read_excel(ref_file)

    print("正在读取待测化合物文件...")
    test_df = pd.read_excel(test_file)

    # 检查列名
    if smiles_col not in ref_df.columns:
        raise ValueError(
            f"参考文件中未找到列名：{smiles_col}，当前列名为：{ref_df.columns.tolist()}"
        )

    if smiles_col not in test_df.columns:
        raise ValueError(
            f"待测文件中未找到列名：{smiles_col}，当前列名为：{test_df.columns.tolist()}"
        )

    if id_col not in test_df.columns:
        raise ValueError(
            f"待测文件中未找到 ID 列：{id_col}，当前列名为：{test_df.columns.tolist()}"
        )

    print("正在计算参考化合物 Morgan 指纹...")
    ref_data = prepare_reference_fps(
        ref_df=ref_df,
        smiles_col=smiles_col,
        radius=RADIUS,
        n_bits=N_BITS
    )

    print(f"参考化合物总数：{len(ref_df)}")
    print(f"有效参考化合物数量：{len(ref_data)}")

    results = []

    print("开始计算 Morgan Tanimoto 相似度...")

    for idx, row in test_df.iterrows():
        query_smiles = row[smiles_col]
        compound_id = row[id_col]          # 获取 ID

        res = calculate_top5_morgan_tanimoto(
            query_smiles=query_smiles,
            ref_data=ref_data,
            top_n=TOP_N,
            radius=RADIUS,
            n_bits=N_BITS
        )

        res["compound_id"] = compound_id   # 添加 ID 字段
        results.append(res)

        if (idx + 1) % 50 == 0:
            print(f"已处理 {idx + 1}/{len(test_df)} 个待测化合物")

    result_df = pd.DataFrame(results)

    # 调整输出列顺序，ID 列放在第一位
    first_cols = [
        "compound_id",
        "test_smiles",
        "top5_mean_similarity",
        "max_similarity",
        "top1_ref_smiles",
        "top1_similarity",
        "top2_ref_smiles",
        "top2_similarity",
        "top3_ref_smiles",
        "top3_similarity",
        "top4_ref_smiles",
        "top4_similarity",
        "top5_ref_smiles",
        "top5_similarity",
        "error"
    ]

    result_df = result_df[first_cols]

    # 按 ID 升序排列（若 ID 为数值则数值排序；若为字符串则按字典序）
    result_df = result_df.sort_values(by="compound_id").reset_index(drop=True)

    result_df.to_excel(output_file, index=False)

    print("\n计算完成！")
    print(f"结果已保存为：{output_file}")

    print("\n结果预览（前10行）：")
    print(result_df.head(10).to_string(index=False))