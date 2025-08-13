#!/usr/bin/env python3
"""
测试双网络模型包装器的修复
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

# 模拟双网络模型
class MockDualNetworkModel:
    def __init__(self):
        self.search_dims = 3
        self.network_input_dims = 4
        self.element_scaler = StandardScaler()
        self.is_trained = True
        
        # 模拟训练数据
        X_train = np.random.rand(100, 4)
        self.element_scaler.fit(X_train)
    
    def predict(self, x):
        return np.random.rand(len(x), 1)
    
    def predict_detailed(self, x):
        return {
            'elastic_modulus': np.random.rand(len(x), 1),
            'yield_strength': np.random.rand(len(x), 1),
            'combined': np.random.rand(len(x), 1)
        }

# 双网络模型包装器
class DualNetworkModelWrapper:
    def __init__(self, dual_model):
        self.dual_model = dual_model
        self.element_scaler = dual_model.element_scaler
        self.input_dims = dual_model.search_dims  # 3维搜索空间
        # 为了兼容性，添加surrogate_instance属性
        self.surrogate_instance = dual_model
        
    def predict(self, x, verbose=0):
        """使用双网络模型进行预测"""
        # 确保输入维度正确
        if isinstance(x, (list, tuple)):
            x = np.array(x)
        
        # 处理维度问题 - DANTE传递的是3D数组 (batch_size, dims, 1)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim == 3:
            if x.shape[2] == 1:
                x = x.squeeze(2)  # 移除最后一个维度
            else:
                x = x.reshape(x.shape[0], -1)
        elif x.ndim > 3:
            x = x.reshape(x.shape[0], -1)
        
        # 确保最终是2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # 检查维度是否正确
        if x.shape[1] != self.input_dims:
            print(f"Warning: Expected {self.input_dims} dimensions, got {x.shape[1]}")
            if x.shape[1] > self.input_dims:
                x = x[:, :self.input_dims]
            elif x.shape[1] < self.input_dims:
                padding = np.zeros((x.shape[0], self.input_dims - x.shape[1]))
                x = np.concatenate([x, padding], axis=1)
        
        try:
            # 使用双网络模型的预测方法
            combined_pred = self.dual_model.predict(x, verbose=verbose)
            return combined_pred
        except Exception as e:
            print(f"Dual network prediction error: {e}")
            print(f"Input shape: {x.shape}")
            raise
    
    def __call__(self, x, y, **kwargs):
        print("使用训练好的双网络模型（弹性模量网络 + 屈服强度网络），跳过训练过程...")
        return self

def test_dual_network_wrapper():
    """测试双网络模型包装器"""
    print("🧪 测试双网络模型包装器修复")
    print("=" * 40)
    
    # 创建模拟的双网络模型
    mock_dual_model = MockDualNetworkModel()
    print("✅ 创建模拟双网络模型")
    
    # 创建包装器
    wrapper = DualNetworkModelWrapper(mock_dual_model)
    print("✅ 创建双网络模型包装器")
    
    # 测试1：检查属性是否存在
    print("\n测试1: 检查必要属性")
    try:
        assert hasattr(wrapper, 'dual_model'), "缺少 dual_model 属性"
        assert hasattr(wrapper, 'element_scaler'), "缺少 element_scaler 属性"
        assert hasattr(wrapper, 'input_dims'), "缺少 input_dims 属性"
        assert hasattr(wrapper, 'surrogate_instance'), "缺少 surrogate_instance 属性"
        print("✅ 所有必要属性都存在")
    except AssertionError as e:
        print(f"❌ 属性检查失败: {e}")
        return False
    
    # 测试2：检查surrogate_instance的属性
    print("\n测试2: 检查surrogate_instance属性")
    try:
        # 这是导致错误的检查
        has_elastic = hasattr(wrapper.surrogate_instance, 'elastic_model')
        has_yield = hasattr(wrapper.surrogate_instance, 'yield_model')
        print(f"surrogate_instance.elastic_model: {has_elastic}")
        print(f"surrogate_instance.yield_model: {has_yield}")
        
        # 检查predict_detailed方法
        has_predict_detailed = hasattr(wrapper.surrogate_instance, 'predict_detailed')
        print(f"surrogate_instance.predict_detailed: {has_predict_detailed}")
        
        if has_predict_detailed:
            print("✅ surrogate_instance属性检查通过")
        else:
            print("⚠️ surrogate_instance缺少predict_detailed方法")
    except Exception as e:
        print(f"❌ surrogate_instance属性检查失败: {e}")
        return False
    
    # 测试3：测试预测功能
    print("\n测试3: 测试预测功能")
    try:
        test_input = np.array([[9.0, 5.0, 2.0]])  # 3维输入
        result = wrapper.predict(test_input)
        print(f"✅ 预测成功，结果形状: {result.shape}")
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        return False
    
    # 测试4：测试详细预测功能
    print("\n测试4: 测试详细预测功能")
    try:
        if hasattr(wrapper, 'dual_model') and hasattr(wrapper.dual_model, 'predict_detailed'):
            detailed_result = wrapper.dual_model.predict_detailed(test_input)
            print(f"✅ 详细预测成功，包含键: {list(detailed_result.keys())}")
        else:
            print("⚠️ 详细预测方法不可用")
    except Exception as e:
        print(f"❌ 详细预测失败: {e}")
        return False
    
    # 测试5：模拟DANTE调用
    print("\n测试5: 模拟DANTE调用")
    try:
        # 模拟DANTE的调用方式
        model = wrapper(None, None)  # DANTE会这样调用
        print("✅ DANTE调用模拟成功")
    except Exception as e:
        print(f"❌ DANTE调用模拟失败: {e}")
        return False
    
    print("\n🎉 所有测试通过！双网络模型包装器修复成功。")
    return True

if __name__ == "__main__":
    success = test_dual_network_wrapper()
    if success:
        print("\n✅ 双网络模型包装器修复验证成功！")
        print("现在应该不会再出现 'surrogate_instance' 属性错误。")
    else:
        print("\n❌ 双网络模型包装器仍有问题，需要进一步调试。")
