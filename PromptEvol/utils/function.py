import numpy as np

def batch_softmax(x, axis=-1):
    """
    支持任意维度的批量 softmax
    """
    # 确保 axis 为正数
    if axis < 0:
        axis = x.ndim + axis
    
    # 重塑以便向量化计算
    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[axis]) if x.ndim > 1 else x.reshape(1, -1)

    # 计算 softmax
    # x_max = np.max(x_2d, axis=1, keepdims=True)
    exp_x = np.exp(x_2d)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    probs_2d = exp_x / sum_exp_x
    
    # 恢复原始形状
    return probs_2d.reshape(original_shape)