import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from collections import Counter


class GravitationalClassifier:
    """
    引力分类器 (Gravitational Classifier)
    基于万有引力定律的分类算法，使用节点度和局部密度定义质量，并计算引力作用力。

    针对极度不平衡数据集增强的功能：   
    - 引力计算中的类别权重调整
    - G-Mean优化阈值
    - 可配置的平衡比率
    """

    def __init__(self, **kwargs):
        """
        初始化引力分类器
        
        参数说明:
        - G: 引力常数，控制整体引力强度
        - k_neighbors: K近邻数，用于构建类内图结构
        - mass_weights: 质量计算权重 [节点度权重, 密度权重]
        - distance_metric: 距离度量方式 ('euclidean' 或 'cosine')
        - gravity_exponent: 引力指数，通常为2
        - gravity_weight: KNN引力与全局引力的混合权重
        
        不平衡数据优化参数:
        - 使用类别权重
        """
        self.X_train = None          # 训练特征数据
        self.y_train = None          # 训练标签数据
        self.class_labels = None     # 类别标签列表
        self.graph_structures = None # 各类别的图结构
        self.mass_vectors = None     # 各类别的质量向量
        
        # 核心引力参数
        self.G = 1.0                 # 引力常数
        self.k_neighbors = 7         # K近邻数
        self.mass_weights = [1.0, 1.0]  # 质量计算权重 [度, 密度]
        self.distance_metric = 'euclidean'  # 距离度量
        self.gravity_exponent = 2    # 引力指数
        self.gravity_weight = 0.7    # KNN/全局引力混合权重

        # 通过kwargs动态设置参数
        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit(self, X, y):
        """
        训练分类器  

        参数:
        - X: 特征矩阵
        - y: 标签向量

        返回:
        - self: 训练后的分类器实例
        """
        # 保存训练数据
        self.X_train = X
        self.y_train = y

        # 获取类别标签并构建图结构
        self.class_labels = np.unique(y)
        print("正在构建类别图结构并计算质量向量...")
        self.build_class_graphs()
        self.compute_mass_vectors()
        return self
   
    def build_class_graphs(self):
        """
        为每个类别构建图结构
        
        每个类别的图结构包含:
        - adjacency: 邻接矩阵
        - samples: 样本矩阵
        - k_used: 实际使用的K值
        - mean_class_dist: 类内平均距离
        """
        n_classes = len(self.class_labels)
        y_train = self.y_train
        X_train = self.X_train
        class_labels = self.class_labels
        distance_metric = self.distance_metric

        temp_graphs = []
        for i in range(n_classes):
            # 获取当前类别的样本
            class_idx = (y_train == class_labels[i])
            X_class = X_train[class_idx, :]
            n_samples = X_class.shape[0]
            
            # 如果没有样本，创建空图结构
            if n_samples == 0:
                temp_graphs.append({
                    'adjacency': None,
                    'samples': None,
                    'k_used': 0,
                    'mean_class_dist': 0
                })
                continue

            # 计算实际使用的K值
            actual_k = min(self.k_neighbors, n_samples - 1)
            
            # 构建邻接矩阵
            if actual_k < 1:
                adj_matrix = csr_matrix((n_samples, n_samples))
            else:
                nn = NearestNeighbors(n_neighbors=actual_k+1, metric=distance_metric)
                nn.fit(X_class)
                idx = nn.kneighbors(X_class, return_distance=False)               

                # 构建稀疏邻接矩阵
                neighbors = idx[:, 1:]
                row = np.repeat(np.arange(n_samples), actual_k)
                col = neighbors.flatten()
                data = np.ones(len(row))
                adj_matrix = csr_matrix((data, (row, col)), shape=(n_samples, n_samples))
                adj_matrix = adj_matrix.maximum(adj_matrix.T)  # 转为无向图
                adj_matrix.setdiag(0)  # 对角线置零

            # 计算类内平均距离
            if n_samples > 1:               
                dist_local = cdist(X_class, X_class, metric=distance_metric)
                mean_class_dist = np.mean(dist_local)
            else:
                mean_class_dist = 0

            # 保存图结构
            temp_graphs.append({
                'adjacency': adj_matrix,
                'samples': X_class,
                'k_used': actual_k,
                'mean_class_dist': mean_class_dist
            })
            # 获取类别名称，如果有映射则使用原始名称
            class_label = class_labels[i]
            class_name = str(class_label)
            if hasattr(self, 'label_map') and self.label_map is not None:
                if class_label in self.label_map:
                    class_name = self.label_map[class_label]
            
            print(f"类别 {class_name}: 已构建图，样本数={n_samples} (k={actual_k})")
        self.graph_structures = temp_graphs

    def compute_mass_vectors(self):
        """
        计算每个类别的质量向量
        
        质量计算基于:
        - 节点度（图结构中的连接数）
        - 局部密度（K近邻平均距离的倒数）        
        """
        n_classes = len(self.class_labels)
        self.mass_vectors = []
        w = np.array(self.mass_weights)
        w_sum = np.sum(w)
                        
        for i in range(n_classes):
            graph = self.graph_structures[i]
            adj = graph['adjacency']
            # 返回邻接矩阵的行数，也就是该类别的样本数量
            n_samp = adj.shape[0] if adj is not None else 0
            
            # 初始化度和密度数组
            deg = np.zeros(n_samp)
            den = np.zeros(n_samp)
            
            # 计算节点度
            if w[0] > 0 and adj is not None:
                deg = np.asarray(adj.sum(axis=1)).flatten()
            
            # 计算局部密度
            if w[1] > 0 and n_samp > 1:
                # 当前类别的所有样本
                samples = graph['samples']
                dist_local = cdist(samples, samples, metric=self.distance_metric)
                k = min(self.k_neighbors, n_samp-1)
                if k > 0:
                    # 对距离矩阵进行排序，保留K+1个样本（包括自己）
                    sorted_dist = np.sort(dist_local, axis=1)
                    # 计算K近邻的平均距离,这个平均距离越小，说明该样本周围邻居越密集。
                    local_mean_dist = np.mean(sorted_dist[:, 1:k+1], axis=1)
                    den = 1.0 / (local_mean_dist + 1e-12)
                else:
                    # 如果K为0，所有样本的密度都设为1
                    den = np.ones(n_samp)
            elif w[1] > 0 and n_samp == 1:
                den = np.ones(1)
            
            # 计算质量（加权组合度和密度）
            mass = (w[0]*deg + w[1]*den) / w_sum
            
            # 归一化质量向量
            total = np.sum(mass)
            if total > 0:
                mass = mass / total
            else:
                mass = np.ones_like(mass) / len(mass)
            
            self.mass_vectors.append(mass)

    def predict(self, X_test):
        """
        对测试样本进行预测
        
        参数:
        - X_test: 测试特征矩阵
        
        返回:
        - y_pred: 预测标签
        - scores: 各类别的引力分数
        """
        n_test = X_test.shape[0]
        n_classes = len(self.class_labels)
        scores = np.zeros((n_test, n_classes))

        # 计算每个测试样本在各个类别中的质量
        all_test_masses = np.zeros((n_test, n_classes))
        for i in range(n_test):
            all_test_masses[i, :] = self.compute_test_sample_masses(X_test[i, :])

        # 获取训练集类别统计信息 
        class_counts = np.bincount(self.y_train.astype(int))
        total_samples = np.sum(class_counts)
        
        # 对每个类别计算引力分数
        for j in range(n_classes):
            graph = self.graph_structures[j]
            class_samples = graph['samples']
            mass_vec = self.mass_vectors[j]
            n_samples = class_samples.shape[0]
            if n_samples == 0:
                continue

            # 计算测试样本与训练样本的距离
            dist = cdist(X_test, class_samples, metric=self.distance_metric)
            dist = np.maximum(dist, 1e-12)  # 避免除零

            # 获取引力指数
            exp_adj = self.gravity_exponent
            k_eff = graph['k_used']
            # 计算测试样本在当前类别中的质量
            test_masses = all_test_masses[:, j]
            
            # 计算动态类别权重（基于不平衡程度）
            label = self.class_labels[j]
            if label < len(class_counts):
                class_ratio = class_counts[label] / total_samples
                imbalance_ratio = class_counts.max() / (class_counts[label] + 1e-10)
                
                # 针对极端不平衡数据的动态类别权重
                base_weight = 3.5
                if class_counts[label] == class_counts.max():
                    # 多数类：权重随不平衡程度适度降低                    
                    class_weight = base_weight - np.log1p(imbalance_ratio) * 0.1
                else:
                    # 少数类：权重随不平衡程度适度增加
                    if imbalance_ratio > 10:
                        # 极端不平衡：适度增强少数类权重
                        class_weight = base_weight + np.log1p(imbalance_ratio) * 0.4 + (1.0 - class_ratio) * imbalance_ratio * 0.08
                    elif imbalance_ratio > 5:
                        # 中度不平衡：适度增强
                        class_weight = base_weight + np.log1p(imbalance_ratio) * 0.3 + (1.0 - class_ratio) * imbalance_ratio * 0.05
                    else:
                        # 轻度不平衡：轻微增强
                        class_weight = base_weight + np.log1p(imbalance_ratio) * 0.02
                
                # 限制权重范围1.5-6.0，避免过度调整
                class_weight = np.clip(class_weight, 1.5, 6.0)           
          
            print(f"类别 {label}: 权重={class_weight:.2f}")
            # 计算全局引力（所有样本）
            sum_full = np.sum(mass_vec[None, :] / (dist ** exp_adj), axis=1)
            gravity_full = self.G * test_masses * sum_full / n_samples

            # 计算KNN引力（仅K近邻）
            gravity_knn = np.zeros(n_test)
            for i in range(n_test):
                # 排序距离，选择K近邻样本
                d_sorted_indices = np.argsort(dist[i, :])
                top_k = min(k_eff, len(d_sorted_indices))
                selected_masses = mass_vec[d_sorted_indices[:top_k]]
                selected_dists = dist[i, d_sorted_indices[:top_k]]
                gravity_knn[i] = self.G * test_masses[i] * np.sum(selected_masses / (selected_dists ** exp_adj))

            # 混合全局引力和KNN引力
            gravity = self.gravity_weight * gravity_knn + (1 - self.gravity_weight) * gravity_full
            scores[:, j] = gravity * class_weight

        # 初步预测（取最大分数类别）
        # 对每个测试样本，取引力分数最高的类别作为预测
        y_pred_idx = np.argmax(scores, axis=1)
        
        # 将索引转换为实际标签
        y_pred = np.array([self.class_labels[i] for i in y_pred_idx])
        return y_pred, scores

    def compute_test_sample_masses(self, test_sample):
        """
        计算测试样本在各个类别中的质量
        
        参数:
        - test_sample: 单个测试样本（一维数组）
        
        返回:
        - test_masses: 测试样本在各个类别中的质量向量
        """
        test_sample = test_sample.reshape(1, -1)
        n_classes = len(self.class_labels)
        test_masses = np.zeros(n_classes)
        
        for j in range(n_classes):
            graph = self.graph_structures[j]
            class_samples = graph['samples']
            n_samples = class_samples.shape[0]
            
            # 如果类别没有样本，质量设为1.0
            if n_samples == 0:
                test_masses[j] = 1.0
                continue

            k = min(self.k_neighbors, n_samples - 1)
            if k <= 0:
                test_masses[j] = 1.0
                continue

            # 计算测试样本到类别样本的K近邻距离          
            nn = NearestNeighbors(n_neighbors=k, metric=self.distance_metric)
            nn.fit(class_samples)
            dist, _ = nn.kneighbors(test_sample)
            
            # 计算局部密度和全局密度
            local_density = 1.0 / (np.mean(dist) + 1e-12)
            global_density = 1.0 / (graph['mean_class_dist'] + 1e-12)

            # 组合局部密度和全局密度作为测试样本质量
            test_mass = 0.7 * local_density + 0.3 * global_density * n_samples
            avg_mass = np.mean(self.mass_vectors[j]) * n_samples
            
            # 归一化
            if avg_mass > 0:
                test_mass /= avg_mass
            
            test_masses[j] = test_mass
        
        return test_masses
  