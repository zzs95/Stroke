from get_mips_data import get_mips_data
from preprocess import append_csv_features
import numpy as np

# temp = get_mips_data()

# print(len(temp.keys()))

append_csv_features(get_mips_data())