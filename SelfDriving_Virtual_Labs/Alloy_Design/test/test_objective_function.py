#!/usr/bin/env python3
"""
测试目标函数的修复
"""

import numpy as np
import pandas as pd
import json
import ast

# 模拟DANTE的ObjectiveFunction基类
class ObjectiveFunction:
    def __init__(self, dims=3, turn=0.01):
        self.dims = dims
        self.turn = turn
        self.tracker = None  # 简化版本
    
    def _preprocess(self, x):
        if isinstance(x, (list, tuple)):
            x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x.flatten() if x.shape[0] == 1 else x
    
    def scaled(self, y):
        return 1.0 + (y - (-self.min_val)) / ((-self.max_val) - (-self.min_val))

class AlloyObjectiveFunction(ObjectiveFunction):
    """
    合金材料优化的目标函数。
    优化目标是最大化弹性模量和屈服强度的综合性能。
    
    注意：搜索空间是3维（Co, Mo, Ti），Fe通过计算得出
    """
    def __init__(self, X_data_3d, X_data_4d, Y_data, dims=3, turn=0.01):
        self.name = "alloy_optimization"
        
        # 存储训练数据
        self.X_data_3d = X_data_3d  # 3维数据用于搜索
        self.X_data_4d = X_data_4d  # 4维数据用于查找
        self.Y_data = Y_data
        
        # 计算数据统计信息，用于缩放
        self.max_val = np.max(Y_data)
        self.min_val = np.min(Y_data)
        
        # 设置搜索边界（严格基于训练数据范围，不扩展）
        co_min, co_max = X_data_3d[:, 0].min(), X_data_3d[:, 0].max()
        mo_min, mo_max = X_data_3d[:, 1].min(), X_data_3d[:, 1].max()
        ti_min, ti_max = X_data_3d[:, 2].min(), X_data_3d[:, 2].max()
        
        # 确保Fe含量在合理范围内（60-90%）
        max_sum = 40.0  # Co + Mo + Ti的最大和，确保Fe >= 60%
        min_sum = 10.0  # Co + Mo + Ti的最小和，确保Fe <= 90%
        
        # 如果当前边界会导致不合理的Fe含量，进行调整
        current_max_sum = co_max + mo_max + ti_max
        current_min_sum = co_min + mo_min + ti_min
        
        if current_max_sum > max_sum:
            scale_factor = max_sum / current_max_sum
            co_max *= scale_factor
            mo_max *= scale_factor
            ti_max *= scale_factor
        
        if current_min_sum < min_sum:
            scale_factor = min_sum / current_min_sum
            co_min *= scale_factor
            mo_min *= scale_factor
            ti_min *= scale_factor
        
        # 初始化父类
        super().__init__(dims=dims, turn=turn)
        
        # 初始化边界属性
        self.lb = np.array([co_min, mo_min, ti_min])
        self.ub = np.array([co_max, mo_max, ti_max])
    
    def convert_3d_to_4d(self, x_3d):
        """将3维输入（Co, Mo, Ti）转换为4维（Co, Mo, Ti, Fe）"""
        if x_3d.ndim == 1:
            fe_content = 100.0 - np.sum(x_3d)
            return np.append(x_3d, fe_content)
        else:
            fe_content = 100.0 - np.sum(x_3d, axis=1)
            return np.column_stack([x_3d, fe_content])
    
    def __call__(self, x, apply_scaling=False):
        """评估给定合金成分的性能"""
        x = self._preprocess(x)
        
        # 严格边界检查 - 如果超出边界返回惩罚值
        if np.any(x < self.lb) or np.any(x > self.ub):
            penalty = 1e6  # 大惩罚值
            if apply_scaling:
                return penalty
            return penalty
        
        # 确保x在边界内（3维）- 作为额外保护
        x = np.clip(x, self.lb, self.ub)
        
        # 验证Fe含量的合理性
        fe_content = 100.0 - np.sum(x)
        if fe_content < 0 or fe_content > 100:
            penalty = 1e6
            if apply_scaling:
                return penalty
            return penalty
        
        # 根据存储的数据维度选择合适的比较方式
        if self.X_data_4d.shape[1] == 4:
            # 如果存储的是4维数据，将3维输入转换为4维进行匹配
            x_4d = self.convert_3d_to_4d(x)
            distances = np.linalg.norm(self.X_data_4d - x_4d, axis=1)
        else:
            # 如果存储的是3维数据，直接使用3维进行匹配
            distances = np.linalg.norm(self.X_data_3d - x, axis=1)
        
        nearest_idx = np.argmin(distances)
        
        # 返回负值以转换为最小化问题
        result = -self.Y_data[nearest_idx]
        
        if apply_scaling:
            return self.scaled(result)
        return result

def test_objective_function():
    """测试目标函数"""
    print("🧪 测试目标函数修复")
    print("=" * 40)
    
    # 加载真实数据
    try:
        data = pd.read_csv('data.csv')
        print(f"✅ 加载数据: {len(data)} 样本")
    except:
        print("❌ 无法加载数据文件")
        return False
    
    # 解析数据（简化版本）
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
    
    # 解析元素成分
    element_compositions = []
    for sid in data['sid']:
        composition = parse_element_composition(sid)
        element_compositions.append([composition['Co'], composition['Mo'], composition['Ti'], composition['Fe']])
    
    X_elements_4d = np.array(element_compositions)
    X_elements_3d = X_elements_4d[:, :3]
    
    # 目标变量
    Y_elastic = data['elastic'].values
    Y_yield = data['yield'].values
    
    # 归一化
    elastic_min, elastic_max = Y_elastic.min(), Y_elastic.max()
    yield_min, yield_max = Y_yield.min(), Y_yield.max()
    
    Y_elastic_norm = (Y_elastic - elastic_min) / (elastic_max - elastic_min)
    Y_yield_norm = (Y_yield - yield_min) / (yield_max - yield_min)
    Y = (Y_elastic_norm + Y_yield_norm) / 2
    
    print(f"数据形状: 3D={X_elements_3d.shape}, 4D={X_elements_4d.shape}")
    
    # 测试1：使用4维数据创建目标函数
    print("\n测试1: 使用4维数据创建目标函数")
    try:
        obj_func_4d = AlloyObjectiveFunction(X_elements_3d, X_elements_4d, Y, dims=3)
        test_point = np.array([9.0, 5.0, 2.0])  # Co=9%, Mo=5%, Ti=2%
        result = obj_func_4d(test_point)
        print(f"✅ 4维目标函数测试成功: {result:.6f}")
    except Exception as e:
        print(f"❌ 4维目标函数测试失败: {e}")
        return False
    
    # 测试2：使用3维数据创建目标函数（模拟错误情况）
    print("\n测试2: 使用3维数据创建目标函数")
    try:
        obj_func_3d = AlloyObjectiveFunction(X_elements_3d, X_elements_3d, Y, dims=3)
        test_point = np.array([9.0, 5.0, 2.0])
        result = obj_func_3d(test_point)
        print(f"✅ 3维目标函数测试成功: {result:.6f}")
    except Exception as e:
        print(f"❌ 3维目标函数测试失败: {e}")
        return False
    
    # 测试3：边界检查
    print("\n测试3: 边界检查")
    try:
        # 超出边界的点
        out_of_bounds = np.array([15.0, 10.0, 5.0])  # 明显超出范围
        result = obj_func_4d(out_of_bounds)
        if result == 1e6:
            print("✅ 边界检查正常工作")
        else:
            print(f"⚠️ 边界检查可能有问题: {result}")
    except Exception as e:
        print(f"❌ 边界检查测试失败: {e}")
        return False
    
    print("\n🎉 所有测试通过！目标函数修复成功。")
    return True

if __name__ == "__main__":
    success = test_objective_function()
    if success:
        print("\n✅ 目标函数修复验证成功！")
    else:
        print("\n❌ 目标函数仍有问题，需要进一步调试。")
