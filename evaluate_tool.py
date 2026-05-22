import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, matthews_corrcoef)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
import warnings

class EvaluateTool:
    """
    EvaluateTool - 执行K折交叉验证并计算各项指标的平均值
    
    支持的指标：准确率、宏平均F1、G-Mean、平均MCC、宏平均召回率、宏平均特异度、宏平均精确率
    
    用法：
        cv = EvaluateTool(classifier_type='KNN', cv_folds=10, stratified=True)
        cv.validate(X, y)
        cv.display()
        mean_metrics = cv.meanMetrics
    """
    
    def __init__(self, **kwargs):
        """
        构造函数，可选参数以键值对形式传入
        
        Args:
            class: 分类器类型，用于选择内置模型或自定义分类器 ('KNN', 'SVM', 'MLP', 或自定义)
            classifier: 分类器对象，必须具有 fit(X,y) 和 predict(X) 方法，predict返回 (y_pred, scores)
            cv_folds: 折数，默认10
            stratified: 是否分层，默认True
            dsfilename: 数据集文件名（用于保存结果）
            useParallel: 是否启用并行计算（默认False）
        """
        # 默认参数
        self.classifier_type = 'KNN'
        self.classifier = None
        self.cv_folds = 10
        self.stratified = True
        self.results = None
        self.meanMetrics = None
        self.stdMetrics = None
        self.dsfilename = None
        self.useParallel = False
        
        # 单次评估指标
        # self.accuracy = 0
        self.macroF1 = 0
        self.gMean = 0
        self.MCC = 0
        self.macroRecall = 0
        self.macroSpecificity = 0
        # self.macroPrecision = 0
        
        # 设置用户提供的参数
        for key, value in kwargs.items():
            if key == 'classifier_type':
                self.classifier_type = value
            elif key == 'class':  # 兼容MATLAB风格的参数名
                self.classifier_type = value
            elif hasattr(self, key):
                setattr(self, key, value)
    
    def compute(self, y_true, y_pred, scores=None):
        """
        计算单次评估的各项指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            scores: 预测分数（用于计算AUC）
        
        Returns:
            self: 返回自身，支持链式调用
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        try:
            y_true = y_true.astype(int)
            y_pred = y_pred.astype(int)
        except (ValueError, TypeError):
            unique_true = np.unique(y_true)
            unique_pred = np.unique(y_pred)
            true_map = {v: i for i, v in enumerate(unique_true)}
            pred_map = {v: i for i, v in enumerate(unique_pred)}
            y_true = np.array([true_map.get(v, 0) for v in y_true])
            y_pred = np.array([pred_map.get(v, 0) for v in y_pred])
        
        # 准确率
        # self.accuracy = accuracy_score(y_true, y_pred)
        
        # 宏平均精确率、召回率、F1
        # self.macroPrecision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        self.macroRecall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        self.macroF1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # 计算混淆矩阵用于G-Mean和特异度
        cm = confusion_matrix(y_true, y_pred)
        num_classes = cm.shape[0]
        
        # 计算各类别的特异度和召回率（灵敏度）
        recalls = []
        specificities = []
        
        for i in range(num_classes):
            # 真阳性
            tp = cm[i, i]
            # 假阴性
            fn = np.sum(cm[i, :]) - tp
            # 假阳性
            fp = np.sum(cm[:, i]) - tp
            # 真阴性
            tn = np.sum(cm) - tp - fn - fp
            
            # 召回率（灵敏度）
            if tp + fn == 0:
                recall_i = 0
            else:
                recall_i = tp / (tp + fn)
            
            # 特异度
            if tn + fp == 0:
                spec_i = 0
            else:
                spec_i = tn / (tn + fp)
            
            recalls.append(recall_i)
            specificities.append(spec_i)
        
        # 宏平均召回率和特异度
        self.macroRecall = np.mean(recalls)
        self.macroSpecificity = np.mean(specificities)
        
        # G-Mean（几何均值）
        if len(recalls) > 0:
            # 几何均值：任意类别召回率为0则整体为0
            recalls_arr = np.array(recalls)
            if np.any(recalls_arr == 0):
                self.gMean = 0.0
            else:
                self.gMean = np.exp(np.mean(np.log(recalls_arr)))
        else:
            self.gMean = 0
        
        # 平均MCC (Matthews Correlation Coefficient)
        try:
            self.MCC = matthews_corrcoef(y_true, y_pred)
        except Exception as e:
            self.MCC = 0.0
            warnings.warn(f"MCC计算失败: {e}")
        
        return self
    
    def _custom_stratified_split(self, y, n_splits, min_fold_samples=1):
        """
        自定义分层分割策略，确保每折各类别均匀分布
        
        Args:
            y: 标签向量
            n_splits: 折数
            min_fold_samples: 每折每个类别最少样本数
        
        Returns:
            splits: 包含(train_idx, test_idx)元组的列表
        """
        unique_classes, class_counts = np.unique(y, return_counts=True)
        n_classes = len(unique_classes)
        
        # 为每个类别分配样本到各折
        fold_assignments = {cls: [] for cls in unique_classes}
        
        for cls, count in zip(unique_classes, class_counts):
            # 获取该类别的所有索引
            cls_indices = np.where(y == cls)[0]
            np.random.seed(42)
            np.random.shuffle(cls_indices)
            
            # 计算每折分配的样本数
            base_count = count // n_splits
            extra_count = count % n_splits
            
            # 分配样本
            start = 0
            for fold in range(n_splits):
                end = start + base_count + (1 if fold < extra_count else 0)
                # 确保每折至少有min_fold_samples个样本
                if end - start < min_fold_samples and fold < n_splits - 1:
                    remaining = min_fold_samples - (end - start)
                    end += remaining
                fold_assignments[cls].append(cls_indices[start:end])
                start = end
        
        # 构建每折的测试集索引
        splits = []
        for fold in range(n_splits):
            test_indices = []
            for cls in unique_classes:
                test_indices.extend(fold_assignments[cls][fold])
            
            test_indices = np.array(test_indices, dtype=np.int64)
            train_indices = np.array([i for i in range(len(y)) if i not in test_indices], dtype=np.int64)
            
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)
            
            splits.append((train_indices, test_indices))
        
        return splits
    
    def _create_classifier(self):
        """创建指定类型的分类器"""
        switcher = {
            'KNN': KNeighborsClassifier(n_neighbors=7, metric='euclidean'),
            'SVM': SVC(kernel='rbf', probability=True),
            'MLP': MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', 
                                max_iter=1000, verbose=False, random_state=42)
        }
        
        if self.classifier_type in switcher:
            return switcher[self.classifier_type]
        elif self.classifier is not None:
            # 自定义分类器：返回具有相同参数的新实例
            return self.classifier.__class__(**{k: v for k, v in vars(self.classifier).items() if not k.startswith('_')})
        else:
            raise ValueError(f"未知的分类器类型: {self.classifier_type}")
    
    def validate(self, X, y):
        """
        执行交叉验证
        
        Args:
            X: 特征矩阵
            y: 标签向量
        
        Returns:
            self: 返回自身，支持链式调用
        """
        np.random.seed(1)
        
        n_samples = X.shape[0]
        y = np.array(y).flatten()
        
        # 检查分层可行性
        if self.stratified:
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_class_size = np.min(class_counts)
            min_fold_samples = getattr(self, 'min_fold_samples', 1)
            
            if min_class_size < self.cv_folds * min_fold_samples:
                warnings.warn(f"最小类别样本数 ({min_class_size}) 小于每折所需样本数 ({self.cv_folds * min_fold_samples})，尝试使用自定义分层策略。")
                # 使用自定义分层策略确保每折各类别均匀分布
                folds = self._custom_stratified_split(y, self.cv_folds, min_fold_samples)
                use_custom_split = True
            else:
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=1)
                use_custom_split = False
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=1)
            use_custom_split = False
        
        # 预分配结果数组
        # acc = np.zeros(self.cv_folds)
        mf1 = np.zeros(self.cv_folds)
        gm = np.zeros(self.cv_folds)
        auc = np.zeros(self.cv_folds)
        recall = np.zeros(self.cv_folds)
        spec = np.zeros(self.cv_folds)
        # prec = np.zeros(self.cv_folds)
        
        fold = 1
        if use_custom_split:
            splits = folds
        else:
            splits = cv.split(X, y)
        
        for train_idx, test_idx in splits:
            print(f"\n======{self.classifier_type}===开始第{fold}/{self.cv_folds}次训练=============")
            
            X_train = X[train_idx, :]
            y_train = y[train_idx]
            X_test = X[test_idx, :]
            y_test = y[test_idx]
            
            # 打印每折的类别分布
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            unique_test, counts_test = np.unique(y_test, return_counts=True)
            print(f"训练样本: {X_train.shape[0]}, 测试样本: {X_test.shape[0]}")
            print(f"训练集类别分布: {dict(zip([int(u) for u in unique_train], [int(c) for c in counts_train]))}")
            print(f"测试集类别分布: {dict(zip([int(u) for u in unique_test], [int(c) for c in counts_test]))}")
            
            # 设置随机种子（基于fold）
            np.random.seed(fold)
            
            # 创建该折的分类器
            clf = self._create_classifier()
            
            # 训练
            clf.fit(X_train, y_train)
            
            # 预测
            print(f"=========开始第{fold}次预测=============")
            if hasattr(clf, 'predict_proba'):
                result = clf.predict(X_test)
                if isinstance(result, tuple) and len(result) == 2:
                    y_pred, scores = result
                else:
                    y_pred = result
                scores = clf.predict_proba(X_test)
            else:
                result = clf.predict(X_test)
                if isinstance(result, tuple) and len(result) == 2:
                    y_pred, scores = result
                else:
                    y_pred = result
                    scores = None
            
            # 计算指标
            ev = EvaluateTool()
            if scores is not None:
                ev.compute(y_test, y_pred, scores)
            else:
                ev.compute(y_test, y_pred)
            
            # 存储指标
            # acc[fold-1] = ev.accuracy
            mf1[fold-1] = ev.macroF1
            gm[fold-1] = ev.gMean
            auc[fold-1] = ev.MCC
            recall[fold-1] = ev.macroRecall
            spec[fold-1] = ev.macroSpecificity
            # prec[fold-1] = ev.macroPrecision
            
            print(f"第 {fold} 折完成，G-mean: {ev.gMean:.4f}  ==== macroF1: {ev.macroF1:.4f}")
            
            fold += 1
        
        # 组装结果结构体
        self.results = {
            # 'accuracy': acc,
            'macroF1': mf1,
            'gMean': gm,
            'MCC': auc,
            'macroRecall': recall,
            'macroSpecificity': spec,
            # 'macroPrecision': prec
        }
        
        # 计算均值和标准差
        self.meanMetrics = {
            # 'accuracy': np.mean(acc),
            'macroF1': np.mean(mf1),
            'gMean': np.mean(gm),
            'MCC': np.mean(auc),
            'macroRecall': np.mean(recall),
            'macroSpecificity': np.mean(spec),
            # 'macroPrecision': np.mean(prec)
        }
        
        self.stdMetrics = {
            # 'accuracy': np.std(acc),
            'macroF1': np.std(mf1),
            'gMean': np.std(gm),
            'MCC': np.std(auc),
            'macroRecall': np.std(recall),
            'macroSpecificity': np.std(spec),
            # 'macroPrecision': np.std(prec)
        }
        
        return self
    
    def display(self):
        """显示交叉验证结果摘要"""
        if self.meanMetrics is None:
            print("尚未执行交叉验证，请先调用 validate(X, y)")
            return
        
        print("\n" + "="*60)
        print(f"{self.classifier_type} - {self.cv_folds}折交叉验证结果摘要")
        print("="*60)
        
        metrics_order = ['macroF1', 'gMean', 'MCC', 
                         'macroRecall', 'macroSpecificity']
        
        metric_names = {
            'macroF1': '宏平均F1',
            'gMean': 'G-Mean',
            'MCC': 'MCC',
            'macroRecall': '宏平均召回率',
            'macroSpecificity': '宏平均特异度',
        }
        
        for metric in metrics_order:
            mean_val = self.meanMetrics[metric]
            std_val = self.stdMetrics[metric]
            print(f"{metric_names[metric]:<12} | 均值: {mean_val:.4f} | 标准差: {std_val:.4f}")
        
        print("="*60)
        
        # 打印每折详情
        print("\n各折详细结果:")
        print("-"*60)
        header = "折数 | 宏F1  | G-Mean | MCC   | 召回率 | 特异度"
        print(header)
        print("-"*60)
        
        for i in range(self.cv_folds):
            print(f"{i+1:4d} | "
                  f"{self.results['macroF1'][i]:.4f} | "
                  f"{self.results['gMean'][i]:.4f} | "
                  f"{self.results['MCC'][i]:.4f} | "
                  f"{self.results['macroRecall'][i]:.4f} | "
                  f"{self.results['macroSpecificity'][i]:.4f}")
        
        print("-"*60)

# 测试示例
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # 生成测试数据
    X, y = make_classification(n_samples=500, n_features=20, n_classes=2, 
                               weights=[0.7, 0.3], random_state=42)
    
    # 测试自定义分类器（GravitationalClassifier）
    try:
        from AGC import GravitationalClassifier
        
        print("测试 GravitationalClassifier:")
        gc = GravitationalClassifier()
        cv = EvaluateTool(classifier_type='Gravitational', classifier=gc, cv_folds=10, stratified=True)
        cv.validate(X, y)
        cv.display()
    except ImportError:
        print("AGC模块未找到，测试内置KNN分类器")
        
        # 测试内置KNN分类器
        print("\n测试 KNN 分类器:")
        cv = EvaluateTool(classifier_type='KNN', cv_folds=10, stratified=True)
        cv.validate(X, y)
        cv.display()