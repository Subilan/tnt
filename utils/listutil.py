def flatten(arr):
    """
    将任意嵌套的列表展平为一维列表，类似 JavaScript 的 Array.prototype.flat(Infinity)

    Args:
        arr: 可能嵌套的列表（或其他可迭代对象，但非字符串/bytes）

    Returns:
        list: 展平后的一维列表
    """
    result = []
    for item in arr:
        # 我们不希望把 "abc" 拆成 ['a','b','c']
        if isinstance(item, (list, tuple)) and not isinstance(item, (str, bytes)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
