#!/usr/bin/env python3
"""
测试改进的双网络模型
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import json
import ast

# 检查TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    from keras.callbacks import EarlyStopping
    print("✅ TensorFlow已加载")
except ImportError:
    print("❌ TensorFlow未安装")
    exit(1)

# GPU内存优化
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU内存优化已启用")
    except:
        print("⚠️ GPU配置失败")

class ImprovedDualNetworkModel:
    """改进的双网络模型"""
    
    def __init__(self, search_dims=3, network_input_dims=4, n_folds=3):
        self.search_dims = search_dims
        self.network_input_dims = network_input_dims
        self.element_scaler = StandardScaler()
        self.n_folds = n_folds
        self.elastic_models = []
        self.yield_models = []
        self.is_trained = False
    
    def convert_3d_to_4d(self, x_3d):
        """将3维输入转换为4维"""
        if x_3d.ndim == 1:
            fe_content = 100.0 - np.sum(x_3d)
            return np.append(x_3d, fe_content)
        else:
            fe_content = 100.0 - np.sum(x_3d, axis=1)
            return np.column_stack([x_3d, fe_content])
    
    def create_neural_network(self, input_dim, output_dim=1, network_type="elastic"):
        """创建改进的神经网络"""
        if network_type == "yield":
            # 屈服强度网络：更深的网络和更小的学习率
            model = keras.Sequential([
                layers.Dense(256, activation='relu', input_shape=(input_dim,)),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.1),
                
                layers.Dense(32, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.1),
                
                layers.Dense(output_dim, activation='linear')
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                loss='mse',
                metrics=['mae']
            )
        else:
            # 弹性模量网络：原始架构
            model = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(input_dim,)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                
                layers.Dense(32, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.1),
                
                layers.Dense(output_dim, activation='linear')
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        
        return model
    
    def train(self, x_search, y_elastic, y_yield, verbose=1):
        """训练改进的双网络模型"""
        x_elements_4d = self.convert_3d_to_4d(x_search)
        x_elements_scaled = self.element_scaler.fit_transform(x_elements_4d)
        
        if verbose:
            print(f"开始改进双网络训练...")
            print(f"弹性模量范围: {y_elastic.min():.4f} - {y_elastic.max():.4f}")
            print(f"屈服强度范围: {y_yield.min():.4f} - {y_yield.max():.4f}")
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        elastic_scores = []
        yield_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(x_elements_scaled)):
            if verbose:
                print(f"训练第 {fold+1}/{self.n_folds} 折...")
            
            X_train, X_val = x_elements_scaled[train_idx], x_elements_scaled[val_idx]
            y_elastic_train, y_elastic_val = y_elastic[train_idx], y_elastic[val_idx]
            y_yield_train, y_yield_val = y_yield[train_idx], y_yield[val_idx]
            
            # 训练弹性模量网络
            elastic_model = self.create_neural_network(
                input_dim=self.network_input_dims, 
                network_type="elastic"
            )
            
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=20, restore_best_weights=True
            )
            
            elastic_model.fit(
                X_train, y_elastic_train,
                validation_data=(X_val, y_elastic_val),
                epochs=500,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # 训练屈服强度网络（改进版）
            yield_model = self.create_neural_network(
                input_dim=self.network_input_dims, 
                network_type="yield"
            )
            
            early_stopping_yield = EarlyStopping(
                monitor='val_loss', patience=30, restore_best_weights=True
            )
            
            yield_model.fit(
                X_train, y_yield_train,
                validation_data=(X_val, y_yield_val),
                epochs=800,  # 更多epochs
                batch_size=16,  # 更小batch size
                callbacks=[early_stopping_yield],
                verbose=0
            )
            
            # 评估
            elastic_pred = elastic_model.predict(X_val, verbose=0)
            yield_pred = yield_model.predict(X_val, verbose=0)
            
            elastic_r2 = r2_score(y_elastic_val, elastic_pred)
            yield_r2 = r2_score(y_yield_val, yield_pred)
            
            elastic_scores.append(elastic_r2)
            yield_scores.append(yield_r2)
            
            if verbose:
                print(f"  弹性模量 R²: {elastic_r2:.4f}")
                print(f"  屈服强度 R²: {yield_r2:.4f}")
            
            self.elastic_models.append(elastic_model)
            self.yield_models.append(yield_model)
            
            # 清理内存
            tf.keras.backend.clear_session()
        
        avg_elastic_r2 = np.mean(elastic_scores)
        avg_yield_r2 = np.mean(yield_scores)
        
        if verbose:
            print(f"\n改进双网络交叉验证结果:")
            print(f"弹性模量平均 R²: {avg_elastic_r2:.4f} ± {np.std(elastic_scores):.4f}")
            print(f"屈服强度平均 R²: {avg_yield_r2:.4f} ± {np.std(yield_scores):.4f}")
        
        self.is_trained = True
        return self
    
    def predict_detailed(self, x):
        """详细预测"""
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        x_elements_4d = self.convert_3d_to_4d(x)
        x_elements_scaled = self.element_scaler.transform(x_elements_4d)
        
        # 集成预测
        elastic_predictions = []
        for model in self.elastic_models:
            pred = model.predict(x_elements_scaled, verbose=0)
            elastic_predictions.append(pred)
        elastic_mean = np.mean(elastic_predictions, axis=0)
        
        yield_predictions = []
        for model in self.yield_models:
            pred = model.predict(x_elements_scaled, verbose=0)
            yield_predictions.append(pred)
        yield_mean = np.mean(yield_predictions, axis=0)
        
        return {
            'elastic_modulus': elastic_mean,
            'yield_strength': yield_mean,
            'combined': (elastic_mean + yield_mean) / 2
        }

def test_improved_dual_network():
    """测试改进的双网络模型"""
    print("🧪 测试改进的双网络模型")
    print("=" * 50)
    
    # 加载真实数据
    try:
        data = pd.read_csv('data.csv')
        print(f"✅ 加载数据: {len(data)} 样本")
    except:
        print("❌ 无法加载数据文件")
        return False
    
    # 数据预处理（简化版）
    def parse_element_composition(sid):
        elements = {'Co': 0, 'Mo': 0, 'Ti': 0, 'Fe': 0}
        sid = sid.replace('Co', ' Co').replace('Mo', ' Mo').replace('Ti', ' Ti').replace('Fe', ' Fe').strip()
        parts = sid.split()
        for part in parts:
            if 'Co' in part:
                elements['Co'] = float(part.replace('Co', ''))
            elif 'Mo' in part:
                elements['Mo'] = float(part.replace('Mo', ''))
            elif 'Ti' in part:
                elements['Ti'] = float(part.replace('Ti', ''))
            elif 'Fe' in part:
                elements['Fe'] = float(part.replace('Fe', ''))
        
        if elements['Fe'] == 0:
            elements['Fe'] = 100.0 - (elements['Co'] + elements['Mo'] + elements['Ti'])
        
        return elements
    
    # 解析数据
    element_compositions = []
    for sid in data['sid']:
        composition = parse_element_composition(sid)
        element_compositions.append([composition['Co'], composition['Mo'], composition['Ti']])
    
    X_elements_3d = np.array(element_compositions)
    
    # 目标变量
    Y_elastic_raw = data['elastic'].values
    Y_yield_raw = data['yield'].values
    
    # 归一化
    elastic_min, elastic_max = Y_elastic_raw.min(), Y_elastic_raw.max()
    yield_min, yield_max = Y_yield_raw.min(), Y_yield_raw.max()
    
    Y_elastic = (Y_elastic_raw - elastic_min) / (elastic_max - elastic_min)
    Y_yield = (Y_yield_raw - yield_min) / (yield_max - yield_min)
    
    print(f"数据形状: {X_elements_3d.shape}")
    print(f"弹性模量归一化范围: {Y_elastic.min():.4f} - {Y_elastic.max():.4f}")
    print(f"屈服强度归一化范围: {Y_yield.min():.4f} - {Y_yield.max():.4f}")
    
    # 训练改进的双网络模型
    model = ImprovedDualNetworkModel(search_dims=3, network_input_dims=4, n_folds=3)
    
    try:
        trained_model = model.train(X_elements_3d, Y_elastic, Y_yield, verbose=1)
        print("✅ 改进双网络模型训练完成")
        
        # 测试预测
        detailed_pred = trained_model.predict_detailed(X_elements_3d)
        elastic_pred = detailed_pred['elastic_modulus'].flatten()
        yield_pred = detailed_pred['yield_strength'].flatten()
        
        # 计算整体性能
        elastic_r2_overall = r2_score(Y_elastic, elastic_pred)
        yield_r2_overall = r2_score(Y_yield, yield_pred)
        
        print(f"\n整体预测性能:")
        print(f"弹性模量 R²: {elastic_r2_overall:.4f}")
        print(f"屈服强度 R²: {yield_r2_overall:.4f}")
        
        # 创建对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 弹性模量
        ax1.scatter(Y_elastic, elastic_pred, alpha=0.6, s=30, color='blue')
        min_e = min(Y_elastic.min(), elastic_pred.min())
        max_e = max(Y_elastic.max(), elastic_pred.max())
        ax1.plot([min_e, max_e], [min_e, max_e], 'r--', linewidth=2)
        ax1.set_xlabel('Actual Elastic Modulus (Normalized)')
        ax1.set_ylabel('Predicted Elastic Modulus (Normalized)')
        ax1.set_title(f'Improved Elastic Modulus Network\nR² = {elastic_r2_overall:.4f}')
        ax1.grid(True, alpha=0.3)
        
        # 屈服强度
        ax2.scatter(Y_yield, yield_pred, alpha=0.6, s=30, color='green')
        min_y = min(Y_yield.min(), yield_pred.min())
        max_y = max(Y_yield.max(), yield_pred.max())
        ax2.plot([min_y, max_y], [min_y, max_y], 'r--', linewidth=2)
        ax2.set_xlabel('Actual Yield Strength (Normalized)')
        ax2.set_ylabel('Predicted Yield Strength (Normalized)')
        ax2.set_title(f'Improved Yield Strength Network\nR² = {yield_r2_overall:.4f}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('improved_dual_network_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 改进效果测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_improved_dual_network()
    if success:
        print("\n🎉 改进的双网络模型测试成功！")
        print("屈服强度网络的预测性能应该有显著提升。")
    else:
        print("\n❌ 测试失败")
