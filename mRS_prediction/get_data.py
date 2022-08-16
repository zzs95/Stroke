import csv
import os
import shutil

file_names = set()

def get_names(filename):
    with open(filename, encoding="utf-8", errors="replace") as file:
        reader = csv.reader(file)
        line = 0
        for row in reader:
            if line != 0:
                name = row[7]
                if name not in file_names:
                    file_names.add("./mips/"+name)
            line+=1

def get_files():

    for filename in file_names:
        try:
            shutil.copytree(filename, "data"+filename.replace(".", ""))
        except:
            print("File not found:")
            print(filename)

get_names("brown_elvo_cleaned_2_23.csv")
print("The following files are copied:")
print(file_names)
get_files()
print("finished")