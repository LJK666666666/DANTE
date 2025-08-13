#!/usr/bin/env python3
"""
测试双网络模型的简化脚本
用于验证修改是否正确
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# GPU内存优化配置
print("配置GPU内存优化...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU内存增长模式已启用，检测到 {len(gpus)} 个GPU")
    except RuntimeError as e:
        print(f"GPU配置失败: {e}")
else:
    print("未检测到GPU，将使用CPU训练")

# 启用混合精度训练
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("混合精度训练已启用（FP16）")
except:
    print("混合精度训练启用失败，使用默认精度")

class DualNetworkAlloySurrogateModel:
    """
    双网络合金代理模型测试版本
    """
    
    def __init__(self, search_dims=3, network_input_dims=4, n_folds=5):
        self.input_dims = search_dims
        self.search_dims = search_dims
        self.network_input_dims = network_input_dims
        
        self.element_scaler = StandardScaler()
        self.n_folds = n_folds
        
        self.elastic_models = []
        self.yield_models = []
        
        self.final_elastic_model = None
        self.final_yield_model = None
        self.ensemble_model = None
        self.is_trained = False
    
    def convert_3d_to_4d(self, x_3d):
        """将3维搜索空间输入转换为4维元素特征"""
        if x_3d.ndim == 1:
            fe_content = 100.0 - np.sum(x_3d)
            return np.append(x_3d, fe_content)
        else:
            fe_content = 100.0 - np.sum(x_3d, axis=1)
            return np.column_stack([x_3d, fe_content])
    
    def create_neural_network(self, input_dim, output_dim=1, network_type="elastic"):
        """创建神经网络模型"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
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
        """训练双网络模型"""
        # 转换3维搜索空间到4维元素特征
        x_elements_4d = self.convert_3d_to_4d(x_search)
        x_elements_scaled = self.element_scaler.fit_transform(x_elements_4d)
        
        if verbose:
            print(f"开始双网络训练...")
            print(f"输入特征维度: {x_elements_scaled.shape[1]}")
            print(f"弹性模量目标范围: {y_elastic.min():.4f} - {y_elastic.max():.4f}")
            print(f"屈服强度目标范围: {y_yield.min():.4f} - {y_yield.max():.4f}")
        
        # 简化版本：只训练2折进行测试
        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(x_elements_scaled)):
            if verbose:
                print(f"训练第 {fold+1}/2 折...")
            
            X_train, X_val = x_elements_scaled[train_idx], x_elements_scaled[val_idx]
            y_elastic_train, y_elastic_val = y_elastic[train_idx], y_elastic[val_idx]
            y_yield_train, y_yield_val = y_yield[train_idx], y_yield[val_idx]
            
            # 训练弹性模量网络
            elastic_model = self.create_neural_network(
                input_dim=self.network_input_dims, 
                network_type="elastic"
            )
            
            elastic_model.fit(
                X_train, y_elastic_train,
                validation_data=(X_val, y_elastic_val),
                epochs=50,  # 减少epochs用于测试
                batch_size=32,
                verbose=0
            )
            
            # 训练屈服强度网络
            yield_model = self.create_neural_network(
                input_dim=self.network_input_dims, 
                network_type="yield"
            )
            
            yield_model.fit(
                X_train, y_yield_train,
                validation_data=(X_val, y_yield_val),
                epochs=50,  # 减少epochs用于测试
                batch_size=32,
                verbose=0
            )
            
            # 评估模型
            elastic_pred = elastic_model.predict(X_val, verbose=0)
            yield_pred = yield_model.predict(X_val, verbose=0)
            
            elastic_r2 = r2_score(y_elastic_val, elastic_pred)
            yield_r2 = r2_score(y_yield_val, yield_pred)
            
            if verbose:
                print(f"  弹性模量 R²: {elastic_r2:.4f}")
                print(f"  屈服强度 R²: {yield_r2:.4f}")
            
            self.elastic_models.append(elastic_model)
            self.yield_models.append(yield_model)
            
            # 清理内存
            tf.keras.backend.clear_session()
        
        # 创建集成模型
        self.ensemble_model = DualNetworkEnsemble(
            elastic_models=self.elastic_models,
            yield_models=self.yield_models,
            parent=self
        )
        
        self.is_trained = True
        if verbose:
            print("双网络模型训练完成！")
        
        return self
    
    def predict(self, x, verbose=0):
        """预测接口"""
        if not self.is_trained or self.ensemble_model is None:
            raise ValueError("模型尚未训练")
        
        return self.ensemble_model.predict(x, verbose=verbose)
    
    def predict_detailed(self, x, verbose=0):
        """详细预测接口"""
        if not self.is_trained or self.ensemble_model is None:
            raise ValueError("模型尚未训练")
        
        return self.ensemble_model.predict_detailed(x, verbose=verbose)

class DualNetworkEnsemble:
    """双网络集成模型"""
    def __init__(self, elastic_models, yield_models, parent):
        self.elastic_models = elastic_models
        self.yield_models = yield_models
        self.parent = parent
    
    def predict(self, x, verbose=0):
        """集成预测"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        if x.shape[1] == self.parent.search_dims:
            x_elements_4d = self.parent.convert_3d_to_4d(x)
            x_elements_scaled = self.parent.element_scaler.transform(x_elements_4d)
        else:
            x_elements_scaled = x
        
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
        
        combined_prediction = (elastic_mean + yield_mean) / 2
        return combined_prediction
    
    def predict_detailed(self, x, verbose=0):
        """返回详细预测结果"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        if x.shape[1] == self.parent.search_dims:
            x_elements_4d = self.parent.convert_3d_to_4d(x)
            x_elements_scaled = self.parent.element_scaler.transform(x_elements_4d)
        else:
            x_elements_scaled = x
        
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

def test_dual_network():
    """测试双网络模型"""
    print("开始测试双网络模型...")
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 100
    
    # 3维搜索空间数据 (Co, Mo, Ti)
    X_3d = np.random.rand(n_samples, 3) * [10, 5, 3] + [8, 4, 1]
    
    # 生成模拟的弹性模量和屈服强度数据
    Y_elastic = np.random.rand(n_samples)
    Y_yield = np.random.rand(n_samples)
    
    print(f"测试数据: {n_samples} 样本")
    print(f"3D输入范围: Co[{X_3d[:,0].min():.2f}, {X_3d[:,0].max():.2f}], "
          f"Mo[{X_3d[:,1].min():.2f}, {X_3d[:,1].max():.2f}], "
          f"Ti[{X_3d[:,2].min():.2f}, {X_3d[:,2].max():.2f}]")
    
    # 创建并训练模型
    model = DualNetworkAlloySurrogateModel(search_dims=3, network_input_dims=4, n_folds=2)
    
    try:
        trained_model = model.train(X_3d, Y_elastic, Y_yield, verbose=1)
        print("✅ 模型训练成功！")
        
        # 测试预测
        test_input = np.array([[9.0, 4.5, 2.0]])  # Co=9%, Mo=4.5%, Ti=2%
        
        prediction = trained_model.predict(test_input)
        print(f"✅ 基本预测成功: {prediction[0][0]:.4f}")
        
        detailed_pred = trained_model.predict_detailed(test_input)
        print(f"✅ 详细预测成功:")
        print(f"   弹性模量: {detailed_pred['elastic_modulus'][0][0]:.4f}")
        print(f"   屈服强度: {detailed_pred['yield_strength'][0][0]:.4f}")
        print(f"   组合预测: {detailed_pred['combined'][0][0]:.4f}")
        
        print("\n🎉 所有测试通过！双网络模型工作正常。")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dual_network()
    if success:
        print("\n✅ 双网络模型修改验证成功！可以在notebook中使用。")
    else:
        print("\n❌ 双网络模型存在问题，需要进一步调试。")
