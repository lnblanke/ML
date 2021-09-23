# Load data from pku machine translation dataset
# @Time: 8/2/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: load_data.py

import pathlib

def load_text(path):
    text = path.read_text(encoding = "utf-8")

    lines = text.splitlines()

    return lines

def load_data():
    en_train = load_text(pathlib.Path("../../data/train.seg.en.txt"))
    zh_train = load_text(pathlib.Path("../../data/train.seg.zh.txt"))
    en_dev = load_text(pathlib.Path("../../data/dev.seg.en.txt"))
    zh_dev = load_text(pathlib.Path("../../data/dev.seg.zh.txt"))
    en_test = load_text(pathlib.Path("../../data/test.seg.en.txt"))
    zh_test = load_text(pathlib.Path("../../data/test.seg.zh.txt"))

    return en_train, zh_train, en_dev, zh_dev, en_test, zh_test
