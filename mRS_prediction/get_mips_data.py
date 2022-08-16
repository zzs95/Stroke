import numpy as np
import os
import pandas as pd

# Show the current working directory
os.getcwd()



def get_mips_data():
    df = pd.read_csv("brown_elvo_cleaned_2_23.csv", engine="python")
    available_sham = df["X_ShamAccessionNumber"].to_list()
    data_dict = dict()
    dir = "./data/mips"

    # Read the images from folders and save them to dict
    # with os.scandir("./data/mips") as folders:
    with os.scandir("./data/mips") as folders:
        for folder in folders:
            if folder.name in available_sham:
                dirpath = dir + "/" + folder.name
                with os.scandir(dirpath) as numpydata:
                    for file in numpydata:
                        filepath = dirpath + "/" + file.name
                        loaded = np.load(filepath)
                        data_dict[folder.name] = loaded
    return data_dict
