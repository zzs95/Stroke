import numpy as np
import os
import csv
import re
import tensorflow as tf
from skimage.transform import resize
from get_mips_data import get_mips_data

def parse_if_acute(status):
    status = status.lower()
    if not status:
        # If blank cell
        return [1, 0, 0]
    elif status == "acute":
        return [0, 1, 0]
    else:
        return [0, 0, 1]


def parse_gender(gender):
    return 1 if gender.lower() == "female" else 0


def parse_vessels_locations(vessel, location):
    sentence_len = 25
    if not vessel:
        vessel = ""
    if not location:
        location = ""
    vessel = vessel.lower()
    vessel = re.sub('[^0-9a-zA-Z]+', ' ', vessel)
    location = location.lower()
    concatenated = vessel + " " + location
    output = concatenated.split()
    while len(output) < sentence_len:
        output.append("*UNK*")

    return output


def parse_passes(passes):
    if passes.isnumeric() and "/" in passes:
        passes = passes.split("/")
    else:
        # No data available
        passes = "9"

    output = [int(x) for x in passes]
    return output


def parse_mrs(mrs):
    output = [int(x) for x in mrs]
    return output


mrs_distribution = np.zeros(7)
passes_distribution = np.zeros(10)


def append_csv_features(image_dict):
    csvpath = "brown_elvo_cleaned_2_23.csv"
    test_fraction = 0.2
    with open(csvpath, encoding="utf-8", errors="replace") as file:
        output = []
        labels = []
        reader = csv.reader(file)
        line = 0
        word2num = dict()
        word2num["*UNK*"] = 0
        curr_idx = 1
        for row in reader:
            if line != 0:

                try:
                    data = image_dict[row[7]]
                except KeyError:
                    continue

                data = np.divide(data, np.float32(3071))
                data = tf.expand_dims(data, axis=3)
                flipped_data = data.numpy()
                for i in range(len(flipped_data)):
                    flipped_data[i] = tf.image.\
                        random_flip_left_right(flipped_data[i])

                if_acute = parse_if_acute(row[4])
                old_mrs = int(row[2])
                vessels = row[8]
                location = row[9]
                vessel_and_locations = parse_vessels_locations(vessels,
                                                               location)
                for i, word in enumerate(vessel_and_locations):
                    if word not in word2num:
                        word2num[word] = curr_idx
                        curr_idx += 1

                    vessel_and_locations[i] = word2num[word]

                passes = parse_passes(row[10])
                mrs = parse_mrs(row[3])
                gender = parse_gender(row[5])
                age = int(row[6])

                # if data.shape != (64, 128, 128, 1):
                #     data = resize(data, (64, 128, 128, 1))

                if mrs is not None and data.shape == (64, 256, 256, 1):
                    for m in mrs:
                        for p in passes:
                            mrs_distribution[m] += 1
                            passes_distribution[p] += 1
                            entry = [data, vessel_and_locations]
                            flipped_entry = [flipped_data, vessel_and_locations]
                            entry.extend(if_acute)
                            entry.append(old_mrs)
                            entry.append(gender)
                            entry.append(age)

                            flipped_entry.extend(if_acute)
                            flipped_entry.append(old_mrs)
                            flipped_entry.append(gender)
                            flipped_entry.append(age)

                            output.append(entry)
                            output.append(flipped_entry)
                            labels.append([m, p])
                            labels.append([m, p])

            line += 1

    output = np.array(output)
    labels = np.array(labels)
    indices = np.arange(output.shape[0])
    np.random.shuffle(indices)
    output = output[indices]
    labels = labels[indices]

    train_data = np.array(output[:int(len(output) * (1 - test_fraction))])
    train_labels = np.array(labels[:int(len(output) * (1 - test_fraction))])
    test_data = np.array(output[int(len(output) * (1 - test_fraction)):])
    test_labels = np.array(labels[int(len(output) * (1 - test_fraction)):])
    print(train_data.shape)
    print(train_labels.shape)

    return train_data, train_labels, test_data, test_labels, len(word2num)

append_csv_features(get_mips_data())
