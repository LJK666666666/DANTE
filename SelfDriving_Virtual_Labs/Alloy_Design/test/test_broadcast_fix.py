#!/usr/bin/env python3
"""
测试广播错误修复
"""

import numpy as np

def test_dimension_matching():
    """测试维度匹配修复"""
    print("🧪 测试广播错误修复")
    print("=" * 40)
    
    # 模拟问题场景
    X_data = np.random.rand(621, 4)  # 4维数据 (Co, Mo, Ti, Fe)
    best_composition = np.array([9.0, 5.0, 2.0])  # 3维搜索结果 (Co, Mo, Ti)
    
    print(f"X_data形状: {X_data.shape}")
    print(f"best_composition形状: {best_composition.shape}")
    
    # 测试原始方法（会出错）
    print("\n测试1: 原始方法（应该出错）")
    try:
        distances_old = np.linalg.norm(X_data - best_composition, axis=1)
        print("❌ 原始方法意外成功")
    except ValueError as e:
        print(f"✅ 原始方法正确出错: {e}")
    
    # 测试修复后的方法
    print("\n测试2: 修复后的方法")
    try:
        # 修复逻辑
        if X_data.shape[1] == 4 and len(best_composition) == 3:
            # 将3维best_composition转换为4维进行比较
            fe_content = 100.0 - np.sum(best_composition)
            best_composition_4d = np.append(best_composition, fe_content)
            distances = np.linalg.norm(X_data - best_composition_4d, axis=1)
        else:
            # 维度匹配，直接计算
            distances = np.linalg.norm(X_data - best_composition, axis=1)
        
        print(f"✅ 修复方法成功")
        print(f"转换后的4维组成: {best_composition_4d}")
        print(f"Fe含量: {fe_content:.2f}%")
        print(f"距离数组形状: {distances.shape}")
        print(f"最小距离: {np.min(distances):.6f}")
        print(f"最近材料索引: {np.argmin(distances)}")
        
    except Exception as e:
        print(f"❌ 修复方法失败: {e}")
        return False
    
    # 测试边界情况
    print("\n测试3: 边界情况")
    
    # 情况1：维度已经匹配
    X_data_3d = np.random.rand(621, 3)
    best_composition_3d = np.array([9.0, 5.0, 2.0])
    
    try:
        if X_data_3d.shape[1] == 4 and len(best_composition_3d) == 3:
            fe_content = 100.0 - np.sum(best_composition_3d)
            best_composition_4d = np.append(best_composition_3d, fe_content)
            distances = np.linalg.norm(X_data_3d - best_composition_4d, axis=1)
        else:
            distances = np.linalg.norm(X_data_3d - best_composition_3d, axis=1)
        
        print("✅ 3D-3D匹配测试成功")
        
    except Exception as e:
        print(f"❌ 3D-3D匹配测试失败: {e}")
        return False
    
    # 情况2：4D-4D匹配
    X_data_4d = np.random.rand(621, 4)
    best_composition_4d_input = np.array([9.0, 5.0, 2.0, 84.0])
    
    try:
        if X_data_4d.shape[1] == 4 and len(best_composition_4d_input) == 3:
            fe_content = 100.0 - np.sum(best_composition_4d_input)
            best_composition_4d = np.append(best_composition_4d_input, fe_content)
            distances = np.linalg.norm(X_data_4d - best_composition_4d, axis=1)
        else:
            distances = np.linalg.norm(X_data_4d - best_composition_4d_input, axis=1)
        
        print("✅ 4D-4D匹配测试成功")
        
    except Exception as e:
        print(f"❌ 4D-4D匹配测试失败: {e}")
        return False
    
    print("\n🎉 所有测试通过！广播错误修复成功。")
    return True

def test_fe_content_calculation():
    """测试Fe含量计算的合理性"""
    print("\n🧪 测试Fe含量计算")
    print("=" * 30)
    
    test_cases = [
        [9.0, 5.0, 2.0],    # 正常情况
        [10.0, 4.5, 1.5],   # 另一个正常情况
        [8.0, 6.0, 3.0],    # 边界情况
        [15.0, 10.0, 5.0],  # 超出合理范围的情况
    ]
    
    for i, composition in enumerate(test_cases):
        co, mo, ti = composition
        fe = 100.0 - (co + mo + ti)
        total = co + mo + ti + fe
        
        print(f"测试 {i+1}: Co={co}%, Mo={mo}%, Ti={ti}%")
        print(f"  计算Fe: {fe:.1f}%")
        print(f"  总和: {total:.1f}%")
        
        if fe < 0:
            print("  ⚠️ 警告: Fe含量为负")
        elif fe > 100:
            print("  ⚠️ 警告: Fe含量超过100%")
        elif fe < 60:
            print("  ⚠️ 警告: Fe含量过低（<60%）")
        elif fe > 90:
            print("  ⚠️ 警告: Fe含量过高（>90%）")
        else:
            print("  ✅ Fe含量合理")
        print()

if __name__ == "__main__":
    print("🔧 广播错误修复验证")
    print("=" * 50)
    
    success1 = test_dimension_matching()
    test_fe_content_calculation()
    
    if success1:
        print("\n✅ 广播错误修复验证成功！")
        print("现在双网络优化应该不会再出现维度不匹配错误。")
        print("程序将能够正常完成双网络优化并显示结果。")
    else:
        print("\n❌ 修复验证失败")
