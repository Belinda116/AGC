import numpy as np
import pandas as pd
from AGC import GravitationalClassifier
from evaluate_tool import EvaluateTool

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)

    print("数据集基本信息:")
    print(f"形状: {data.shape}")
    print(f"\n前5行数据:")
    print(data.head())
    print(f"\n各列缺失值统计:")
    print(data.isnull().sum())
    print(f"\n数据类型:")
    print(data.dtypes)

    X_table = data.iloc[:, :-1]
    colNames = X_table.columns.tolist()

    for col in colNames:
        colData = X_table[col].values

        if colData.dtype == 'object' or colData.dtype.name == 'category' or 'string' in str(colData.dtype).lower():
            unique_vals = pd.unique(colData)
            encoding_map = {val: idx+1 for idx, val in enumerate(unique_vals)}
            X_table[col] = X_table[col].map(encoding_map)
        else:
            try:
                if np.issubdtype(colData.dtype, np.number):
                    if np.any(pd.isnull(colData)):
                        col_mean = np.nanmean(colData)
                        colData = np.where(pd.isnull(colData), col_mean, colData)
                        X_table[col] = colData
                else:
                    raise TypeError(f"列 {col} 为不支持的类型 {colData.dtype}")
            except TypeError:
                unique_vals = pd.unique(colData)
                encoding_map = {val: idx+1 for idx, val in enumerate(unique_vals)}
                X_table[col] = X_table[col].map(encoding_map)

    X = X_table.values.astype(float)

    raw_labels = data.iloc[:, -1].values
    
    valid_mask = ~pd.isna(raw_labels)
    X = X[valid_mask, :]
    raw_labels = raw_labels[valid_mask]
    
    label_map = None
    try:
        y = np.asarray(raw_labels).astype(int)
    except (ValueError, TypeError):
        cleaned_labels = np.array([val.strip() if isinstance(val, str) else str(val) for val in raw_labels])
        unique_labels = np.unique(cleaned_labels)
        label_map = {idx+1: val for idx, val in enumerate(unique_labels)}
        label_to_idx = {val: idx+1 for idx, val in enumerate(unique_labels)}
        y = np.array([label_to_idx[val] for val in cleaned_labels])

    print(f"\n处理后 X 形状: {X.shape}")
    print(f"处理后 y 形状: {y.shape}")
    print(f"类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    if label_map:
        print(f"类别名称映射: {label_map}")

    complete_idx = ~np.any(np.isnan(X), axis=1)
    X = X[complete_idx, :]
    y = y[complete_idx]

    print(f"删除缺失值后 X 形状: {X.shape}")
    print(f"删除缺失值后 y 形状: {y.shape}")

    return X, y, label_map


def main():
    dsfilename = 'MHR'
    filepath = f'{dsfilename}.csv'

    print("="*60)
    print(f"加载数据集: {dsfilename}")
    print("="*60)

    try:
        X, y, label_map = load_and_preprocess_data(filepath)
    except FileNotFoundError:
        print(f"\n文件未找到: {filepath}")
        print("使用生成的不平衡数据集进行测试...")
        X, y, label_map = None, None, None


    print("\n" + "="*60)
    print("创建分类器并进行10折交叉验证")
    print("="*60)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    classifier = GravitationalClassifier(
        mass_weights=[0, 1],
        k_neighbors=9,
        distance_metric='euclidean',
        gravity_weight=0.2,
        label_map=label_map
    )
        
    cv = EvaluateTool(
        dsfilename=dsfilename,
        classifier_type='AGC',
        classifier=classifier,
        cv_folds=10,
        stratified=True,
        min_fold_samples=2             
    )

    cv = cv.validate(X, y)
    cv.display()

    print("\n" + "="*60)
    print("测试完成 - 各指标均值:")
    print("="*60)
    for metric, value in cv.meanMetrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # print(f"\n{dsfilename}_{cv.classifier_type}纯数字输出:")
    # metrics_order = ['macroF1', 'gMean', 'MCC', 'macroRecall', 'macroSpecificity']
    # for m in metrics_order:
    #     print(f"{cv.meanMetrics[m]:.4f}")

    print(f"分类器参数: mass_weights=[{classifier.mass_weights[0]}, {classifier.mass_weights[1]}] k_neighbors={classifier.k_neighbors}, distance_metric={classifier.distance_metric}, gravity_weight={classifier.gravity_weight}")


    return cv


if __name__ == "__main__":
    main()