# -*- coding: utf-8 -*-


from os import sys, path
# 得到mini_project所在的绝对路径
package_path = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
# 将mini_project所在路径，加到系统路径里。这样就可以将mini_project作为库使用了
sys.path.append(package_path)
from mini_project.components.selecter import get_frequent_item


def test_selecter():
    data = ["a", "b", "c", "a"]
    re = get_frequent_item(data)
    assert(re[0] == "a")


if __name__ == "__main__":
    print("begin to run test_selecter")
    test_selecter()
