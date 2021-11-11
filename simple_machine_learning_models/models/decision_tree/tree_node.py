# @Time: 9/23/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: tree_node.py

class TreeNode:
    def __init__(self, feature, thres, val, left_child, right_child):
        self.feature = feature
        self.threshold = thres
        self.value = val
        self.left_child = left_child
        self.right_child = right_child
