# -*- coding: utf-8 -*-


from mini_project.components.counter import word_count


def get_frequent_item(data):
    """
    对于给定的字符串列表，找出其中出现次数最多的字符

    参数
    ----
    data: list[str]，字符串列表

    返回
    ----
    re: list[str]，出现次数最多的字符串
    """
    _hash = word_count(data)
    max_num = max(_hash.values())
    return list(filter(lambda x: _hash[x] == max_num, _hash))


if __name__ == "__main__":
    a = ["a", "a", "b", "b", "c"]
    print(get_frequent_item(a))
