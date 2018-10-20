# -*- coding: utf-8 -*-


def word_count(data):
    """
    对于给定的字符串列表，返回各个字符串出现的次数
    
    参数
    ----
    data: list[str]，字符串列表
    
    返回
    ----
    re: dict，返回结果的hash表
    """
    re = {}
    for i in data:
        re[i] = re.get(i, 0) + 1
    return re


if __name__ == "__main__":
    l = ["a", "b", "c"]
    print(word_count(l))