#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SANTOS机器学习评估工具（精简版）
用于评估SANTOS系统通过表格联合(union)增强数据对下游机器学习任务的价值
"""

import argparse
import json
import os
import threading
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, log_loss, precision_score, recall_score, \
    confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class SantosMLEvaluator:
    def __init__(self, query_table_name=None, similiar_table_list=None):
        self.finish = False
        self.thread = None

        """初始化评估器"""
        self.query_table_name = query_table_name
        self.query_table_path, self.query_table = None, None
        if query_table_name is not None:
            self.set_query_table(query_table_name)

        self.similar_tables = similiar_table_list
        if similiar_table_list is not None:
            self.set_similar_tables(similiar_table_list)

        # 生成输出目录（直接使用当前目录）
        self.output_dir = "."
        print(f"\n输出目录: {self.output_dir}")

        # 初始化日志文件
        self.log_file = os.path.join(self.output_dir, 'training_log.json')
        self.progress_data = []

    def set_similar_tables(self, similar_tables):
        """设置相似表格列表"""
        self.similar_tables = similar_tables
        print(f"相似表格列表已更新，当前数量: {len(self.similar_tables)}")

    def set_query_table(self, query_table_name):
        """设置查询表格名称"""
        self.query_table_name = query_table_name
        self.query_table_path = os.path.join("../benchmark/santos_benchmark/query", query_table_name)
        self.query_table = pd.read_csv(self.query_table_path, encoding='latin1')
        print(f"\n数据集信息:")
        print(f"表格名称: {query_table_name}")
        print(f"列数: {len(self.query_table.columns)}")
        print(f"行数: {len(self.query_table)}")

    # def _load_similar_tables(self):
    #     """从CSV文件中加载相似表格列表"""
    #     similar_tables = []
    #     with open(self.true_results_path, 'r', encoding='utf-8') as f:
    #         for line in f:
    #             parts = line.strip().split(',')
    #             if parts[0] == self.query_table_name and parts[1] != self.query_table_name:
    #                 similar_tables.append(parts[1])
    #     return similar_tables

    def _get_model(self, model_type, task_type):
        """根据模型类型和任务类型返回相应的模型"""
        common_params = {
            'random_state': 42
        }

        if task_type == 'classification':
            if model_type == 'decision_tree':
                return DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    **common_params
                )
            elif model_type == 'random_forest':
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features='sqrt',
                    class_weight='balanced',
                    bootstrap=True,
                    oob_score=True,
                    **common_params
                )
            elif model_type == 'gradient_boosting':
                return GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    subsample=0.8,
                    max_depth=5,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    **common_params
                )
            elif model_type == 'svm':
                return SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    probability=True,
                    class_weight='balanced',
                    **common_params
                )
            else:
                raise ValueError(f"不支持的分类模型类型: {model_type}")
        else:
            if model_type == 'decision_tree':
                return DecisionTreeRegressor(
                    max_depth=5,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    **common_params
                )
            elif model_type == 'random_forest':
                return RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features='sqrt',
                    bootstrap=True,
                    oob_score=True,
                    **common_params
                )
            elif model_type == 'gradient_boosting':
                return GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    subsample=0.8,
                    max_depth=5,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    **common_params
                )
            elif model_type == 'svm':
                return SVR(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale'
                )
            else:
                raise ValueError(f"不支持的回归模型类型: {model_type}")

    def prepare_data(self, target_column, task_type='classification', feature_columns=None, k=10, model_type='random_forest'):
        """准备原始数据和增强数据"""
        self.ml_task_type = task_type
        self.target_column = target_column
        self.model_type = model_type

        # 处理特征列
        if feature_columns is None:
            self.feature_columns = [col for col in self.query_table.columns if col != target_column]
        else:
            self.feature_columns = feature_columns

        # 加载相似表格
        similar_tables = self.similar_tables[:k]
        similar_table_dfs = []

        for table_name in similar_tables:
            table_path = os.path.join("../benchmark/santos_benchmark/datalake", table_name)
            if os.path.exists(table_path):
                df = pd.read_csv(table_path, encoding='latin1')
                if target_column in df.columns:
                    selected_columns = [col for col in df.columns if col in self.query_table.columns]
                    if target_column in selected_columns:
                        df = df[selected_columns]
                        if task_type == 'classification':
                            df[target_column] = df[target_column].astype(str)
                            self.query_table[target_column] = self.query_table[target_column].astype(str)
                            original_classes = set(self.query_table[target_column].unique())
                            df = df[df[target_column].isin(original_classes)]
                        similar_table_dfs.append(df)

        # 合并数据
        self.original_data = self.query_table.copy()
        if similar_table_dfs:
            self.augmented_data = pd.concat([self.query_table] + similar_table_dfs, ignore_index=True)
        else:
            self.augmented_data = self.query_table.copy()

        print(f"\n数据增强统计:")
        print(f"原始数据大小: {self.original_data.shape}")
        print(f"增强数据大小: {self.augmented_data.shape}")
        print(f"使用模型: {model_type}")

    def _save_progress(self, epoch, total_epochs, data_type, mode, metrics, cm=None):
        """保存训练进度到日志文件"""
        progress_entry = {
            "type": data_type,
            "mode": mode,
            "epoch": epoch,
            "total_epoch": total_epochs,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics
        }

        if cm is not None:
            progress_entry["confusion_matrix"] = cm.tolist()

        self.progress_data.append(progress_entry)

        # 保存到文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump({"progress": self.progress_data}, f, indent=2, ensure_ascii=False)

    def train_and_evaluate(self, test_size=0.3, n_epochs=10):
        """训练和评估模型"""
        results = {'original': {'train_history': [], 'test_history': []},
                   'augmented': {'train_history': [], 'test_history': []}}

        # 数据预处理
        scaler = StandardScaler()
        label_encoder = LabelEncoder() if self.ml_task_type == 'classification' else None
        feature_encoders = {}

        # 处理原始数据
        X_orig = self.original_data[self.feature_columns].copy()
        y_orig = self.original_data[self.target_column]

        # 对每个特征列进行预处理
        for column in self.feature_columns:
            if X_orig[column].dtype == 'object':  # 如果是字符串类型
                feature_encoders[column] = LabelEncoder()
                X_orig[column] = feature_encoders[column].fit_transform(X_orig[column].astype(str))

        if self.ml_task_type == 'classification':
            y_orig = label_encoder.fit_transform(y_orig.astype(str))

            # 检查类别分布
            class_counts = np.bincount(y_orig)
            min_samples = min(class_counts[class_counts > 0])  # 只考虑非零类别

            if min_samples < 2:
                print(f"警告: 某些类别的样本数太少（最少{min_samples}个），将使用随机抽样代替分层抽样")
                X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
                    X_orig, y_orig, test_size=test_size, random_state=42
                )
            else:
                X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
                    X_orig, y_orig, test_size=test_size, random_state=42, stratify=y_orig
                )
        else:
            X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
                X_orig, y_orig, test_size=test_size, random_state=42
            )

        # 将DataFrame转换为numpy数组
        X_orig_train = X_orig_train.values
        X_orig_test = X_orig_test.values
        y_orig_train = y_orig_train.values if hasattr(y_orig_train, 'values') else y_orig_train
        y_orig_test = y_orig_test.values if hasattr(y_orig_test, 'values') else y_orig_test

        # 训练原始数据模型
        print("\n训练原始数据模型...")
        model_orig = self._get_model(self.model_type, self.ml_task_type)

        # 计算每个epoch的训练数据大小
        min_samples = 30  # 确保每个epoch至少有30个样本
        max_samples = len(X_orig_train)
        subset_sizes = np.linspace(
            min(min_samples, max_samples),
            max_samples,
            n_epochs
        ).astype(int)

        for epoch, train_size in enumerate(subset_sizes, 1):
            # 随机选择训练样本
            indices = np.random.choice(len(X_orig_train), train_size, replace=False)
            X_train_subset = X_orig_train[indices]
            y_train_subset = y_orig_train[indices]

            model_orig.fit(X_train_subset, y_train_subset)

            # 记录训练过程
            if self.ml_task_type == 'classification':
                # 训练集评估
                train_pred = model_orig.predict(X_train_subset)
                train_score = f1_score(y_train_subset, train_pred, average='weighted')
                train_acc = accuracy_score(y_train_subset, train_pred)
                train_pre = precision_score(y_train_subset, train_pred, average='weighted')
                train_rec = recall_score(y_train_subset, train_pred, average='weighted')
                train_cm = confusion_matrix(y_train_subset, train_pred)

                try:
                    classes = np.unique(y_train_subset)
                    train_proba = model_orig.predict_proba(X_train_subset)
                    train_loss = log_loss(y_train_subset, train_proba, labels=classes)
                except Exception as e:
                    print(f"警告: 计算训练损失时出错: {e}")
                    train_loss = 0

                # 保存训练集进度
                train_metrics = {
                    "loss": float(train_loss),
                    "acc": float(train_acc),
                    "pre": float(train_pre),
                    "rec": float(train_rec),
                    "f1": float(train_score)
                }
                self._save_progress(epoch, n_epochs, "query", "train", train_metrics, train_cm)

                # 测试集评估
                test_pred = model_orig.predict(X_orig_test)
                test_score = f1_score(y_orig_test, test_pred, average='weighted')
                test_acc = accuracy_score(y_orig_test, test_pred)
                test_pre = precision_score(y_orig_test, test_pred, average='weighted')
                test_rec = recall_score(y_orig_test, test_pred, average='weighted')
                test_cm = confusion_matrix(y_orig_test, test_pred)

                try:
                    test_proba = model_orig.predict_proba(X_orig_test)
                    test_loss = log_loss(y_orig_test, test_proba, labels=classes)
                except Exception as e:
                    print(f"警告: 计算测试损失时出错: {e}")
                    test_loss = 0

                # 保存测试集进度
                test_metrics = {
                    "loss": float(test_loss),
                    "acc": float(test_acc),
                    "pre": float(test_pre),
                    "rec": float(test_rec),
                    "f1": float(test_score)
                }
                self._save_progress(epoch, n_epochs, "query", "test", test_metrics, test_cm)

            else:
                # 训练集评估
                train_pred = model_orig.predict(X_train_subset)
                train_score = r2_score(y_train_subset, train_pred)
                train_loss = mean_squared_error(y_train_subset, train_pred)

                # 保存训练集进度
                train_metrics = {
                    "loss": float(train_loss),
                    "r2": float(train_score)
                }
                self._save_progress(epoch, n_epochs, "query", "train", train_metrics)

                # 测试集评估
                test_pred = model_orig.predict(X_orig_test)
                test_score = r2_score(y_orig_test, test_pred)
                test_loss = mean_squared_error(y_orig_test, test_pred)

                # 保存测试集进度
                test_metrics = {
                    "loss": float(test_loss),
                    "r2": float(test_score)
                }
                self._save_progress(epoch, n_epochs, "query", "test", test_metrics)

            results['original']['train_history'].append((train_score, train_loss))
            results['original']['test_history'].append((test_score, test_loss))

        # 处理增强数据
        X_aug = self.augmented_data[self.feature_columns].copy()
        y_aug = self.augmented_data[self.target_column]

        # 对增强数据使用相同的编码器进行转换
        for column in self.feature_columns:
            if column in feature_encoders:
                X_aug[column] = X_aug[column].astype(str)
                new_categories = set(X_aug[column].unique()) - set(feature_encoders[column].classes_)
                if new_categories:
                    print(f"警告: 列 '{column}' 中发现新的类别，将被替换为最常见的类别")
                    most_common = feature_encoders[column].classes_[0]
                    X_aug.loc[X_aug[column].isin(new_categories), column] = most_common
                X_aug[column] = feature_encoders[column].transform(X_aug[column])

        if self.ml_task_type == 'classification':
            y_aug = y_aug.astype(str)
            new_categories = set(y_aug.unique()) - set(label_encoder.classes_)
            if new_categories:
                print(f"警告: 目标列中发现新的类别，将被替换为最常见的类别")
                most_common = label_encoder.classes_[0]
                y_aug.loc[y_aug.isin(new_categories)] = most_common
            y_aug = label_encoder.transform(y_aug)

            # 检查类别分布
            class_counts = np.bincount(y_aug)
            min_samples = min(class_counts[class_counts > 0])  # 只考虑非零类别

            if min_samples < 2:
                print(f"警告: 增强数据中某些类别的样本数太少（最少{min_samples}个），将使用随机抽样代替分层抽样")
                X_aug_train, X_aug_test, y_aug_train, y_aug_test = train_test_split(
                    X_aug, y_aug, test_size=test_size, random_state=42
                )
            else:
                X_aug_train, X_aug_test, y_aug_train, y_aug_test = train_test_split(
                    X_aug, y_aug, test_size=test_size, random_state=42, stratify=y_aug
                )
        else:
            X_aug_train, X_aug_test, y_aug_train, y_aug_test = train_test_split(
                X_aug, y_aug, test_size=test_size, random_state=42
            )

        # 将DataFrame转换为numpy数组
        X_aug_train = X_aug_train.values
        X_aug_test = X_aug_test.values
        y_aug_train = y_aug_train.values if hasattr(y_aug_train, 'values') else y_aug_train
        y_aug_test = y_aug_test.values if hasattr(y_aug_test, 'values') else y_aug_test

        # 训练增强数据模型
        print("\n训练增强数据模型...")
        model_aug = self._get_model(self.model_type, self.ml_task_type)

        # 计算每个epoch的训练数据大小
        min_samples = 30
        max_samples = len(X_aug_train)
        subset_sizes = np.linspace(
            min(min_samples, max_samples),
            max_samples,
            n_epochs
        ).astype(int)

        for epoch, train_size in enumerate(subset_sizes, 1):
            # 随机选择训练样本
            indices = np.random.choice(len(X_aug_train), train_size, replace=False)
            X_train_subset = X_aug_train[indices]
            y_train_subset = y_aug_train[indices]

            model_aug.fit(X_train_subset, y_train_subset)

            if self.ml_task_type == 'classification':
                # 训练集评估
                train_pred = model_aug.predict(X_train_subset)
                train_score = f1_score(y_train_subset, train_pred, average='weighted')
                train_acc = accuracy_score(y_train_subset, train_pred)
                train_pre = precision_score(y_train_subset, train_pred, average='weighted')
                train_rec = recall_score(y_train_subset, train_pred, average='weighted')
                train_cm = confusion_matrix(y_train_subset, train_pred)

                try:
                    classes = np.unique(y_train_subset)
                    train_proba = model_aug.predict_proba(X_train_subset)
                    train_loss = log_loss(y_train_subset, train_proba, labels=classes)
                except Exception as e:
                    print(f"警告: 计算训练损失时出错: {e}")
                    train_loss = 0

                # 保存训练集进度
                train_metrics = {
                    "loss": float(train_loss),
                    "acc": float(train_acc),
                    "pre": float(train_pre),
                    "rec": float(train_rec),
                    "f1": float(train_score)
                }
                self._save_progress(epoch, n_epochs, "augmented", "train", train_metrics, train_cm)

                # 测试集评估
                test_pred = model_aug.predict(X_aug_test)
                test_score = f1_score(y_aug_test, test_pred, average='weighted')
                test_acc = accuracy_score(y_aug_test, test_pred)
                test_pre = precision_score(y_aug_test, test_pred, average='weighted')
                test_rec = recall_score(y_aug_test, test_pred, average='weighted')
                test_cm = confusion_matrix(y_aug_test, test_pred)

                try:
                    test_proba = model_aug.predict_proba(X_aug_test)
                    test_loss = log_loss(y_aug_test, test_proba, labels=classes)
                except Exception as e:
                    print(f"警告: 计算测试损失时出错: {e}")
                    test_loss = 0

                # 保存测试集进度
                test_metrics = {
                    "loss": float(test_loss),
                    "acc": float(test_acc),
                    "pre": float(test_pre),
                    "rec": float(test_rec),
                    "f1": float(test_score)
                }
                self._save_progress(epoch, n_epochs, "augmented", "test", test_metrics, test_cm)

            else:
                # 训练集评估
                train_pred = model_aug.predict(X_train_subset)
                train_score = r2_score(y_train_subset, train_pred)
                train_loss = mean_squared_error(y_train_subset, train_pred)

                # 保存训练集进度
                train_metrics = {
                    "loss": float(train_loss),
                    "r2": float(train_score)
                }
                self._save_progress(epoch, n_epochs, "augmented", "train", train_metrics)

                # 测试集评估
                test_pred = model_aug.predict(X_aug_test)
                test_score = r2_score(y_aug_test, test_pred)
                test_loss = mean_squared_error(y_aug_test, test_pred)

                # 保存测试集进度
                test_metrics = {
                    "loss": float(test_loss),
                    "r2": float(test_score)
                }
                self._save_progress(epoch, n_epochs, "augmented", "test", test_metrics)

            results['augmented']['train_history'].append((train_score, train_loss))
            results['augmented']['test_history'].append((test_score, test_loss))

        # 绘制训练过程
        plt.figure(figsize=(15, 5))

        # F1分数曲线
        plt.subplot(1, 2, 1)
        epochs = range(1, n_epochs + 1)

        plt.plot(epochs, [x[0] for x in results['original']['train_history']], 'b-', label='Original (Train)')
        plt.plot(epochs, [x[0] for x in results['original']['test_history']], 'b--', label='Original (Test)')
        plt.plot(epochs, [x[0] for x in results['augmented']['train_history']], 'r-', label='Augmented (Train)')
        plt.plot(epochs, [x[0] for x in results['augmented']['test_history']], 'r--', label='Augmented (Test)')

        metric_name = 'F1 Score' if self.ml_task_type == 'classification' else 'R² Score'
        plt.title(f'Training Process ({metric_name})')
        plt.xlabel('Training Samples')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)

        # Loss曲线
        plt.subplot(1, 2, 2)
        plt.plot(epochs, [x[1] for x in results['original']['train_history']], 'b-', label='Original (Train)')
        plt.plot(epochs, [x[1] for x in results['original']['test_history']], 'b--', label='Original (Test)')
        plt.plot(epochs, [x[1] for x in results['augmented']['train_history']], 'r-', label='Augmented (Train)')
        plt.plot(epochs, [x[1] for x in results['augmented']['test_history']], 'r--', label='Augmented (Test)')

        loss_name = 'Log Loss' if self.ml_task_type == 'classification' else 'MSE'
        plt.title(f'Training Process ({loss_name})')
        plt.xlabel('Training Samples')
        plt.ylabel(loss_name)
        plt.legend()
        plt.grid(True)

        # 保存图表
        plt.tight_layout()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_filename = f"training_process_{self.query_table_name.split('.')[0]}_{self.ml_task_type}_{self.model_type}_{timestamp}.png"
        plt.savefig(os.path.join(self.output_dir, plot_filename))
        plt.close()

        # 打印最终结果
        print("\n最终评估结果:")
        print(f"原始数据 - 测试集{metric_name}: {results['original']['test_history'][-1][0]:.4f}")
        print(f"增强数据 - 测试集{metric_name}: {results['augmented']['test_history'][-1][0]:.4f}")

        improvement = ((results['augmented']['test_history'][-1][0] - results['original']['test_history'][-1][0])
                       / abs(results['original']['test_history'][-1][0]) * 100)
        print(f"性能提升: {improvement:+.2f}%")

        return results

    def run(self, target_column, task_type, model_type):
        def start(target_column, task_type, model_type):
            self.finish = False
            self.prepare_data(target_column=target_column, task_type=task_type, model_type=model_type)
            self.train_and_evaluate()
            self.finish = True

        self.thread = threading.Thread(target=start, args=(target_column, task_type, model_type))
        self.thread.start()

    def get_progress(self):
        return {"finish": self.finish, "progress": self.progress_data}


def main(similier_table_list=None):
    parser = argparse.ArgumentParser(description='SANTOS机器学习评估工具（精简版）')
    parser.add_argument('query_table', nargs='?', default='ydn_spending_data_a.csv',
                        help='查询表格的名称，可以看默认的例子的形式')
    parser.add_argument('--target', '-t', dest='target_column',
                        help='需要分析的目标列名，这个一定得是query表存在的列名')
    parser.add_argument('--task', '-k', dest='task_type', choices=['classification', 'regression'],
                        help='任务类型: classification 或 regression')
    parser.add_argument('--model', '-m', dest='model_type',
                        choices=['decision_tree', 'random_forest', 'gradient_boosting', 'svm'],
                        default='random_forest',
                        help='模型类型: decision_tree, random_forest, gradient_boosting, svm')

    args = parser.parse_args()

    # TODO 这里需要传入一个参数
    evaluator = SantosMLEvaluator(args.query_table, similier_table_list)

    if args.target_column and args.task_type:
        evaluator.prepare_data(target_column=args.target_column,
                               task_type=args.task_type,
                               model_type=args.model_type)
        evaluator.train_and_evaluate()
    else:
        # 默认实验：预测支出类型
        evaluator.prepare_data(target_column='Expense Type',
                               task_type='classification',
                               model_type=args.model_type)
        evaluator.train_and_evaluate()


if __name__ == "__main__":
    main()
    # 本代码运行需要几个参数,日志输出在代码运行目录下面，我感觉就放在codes目录下跑这个代码吧，然后就直接读那个json就好，json文件的名字是固定的叫training_log.json
