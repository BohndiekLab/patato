#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import argparse

import h5py


def init_argparse():
    parser = argparse.ArgumentParser(description="Rename MSOT Scans.")
    parser.add_argument('input_file', type=str, help="Input file name")
    return parser


def main():
    p = init_argparse()
    args = p.parse_args()
    data = h5py.File(args.input_file, "r+")
    name = data.attrs.get("name", None)
    if "name" not in data.attrs:
        name = data["raw_data"].attrs.get("name")
    original_name = data.attrs.get("original_name", None)
    if original_name is None:
        original_name = name
        data.attrs["original_name"] = name
    print(f"Current name: {name}. Original name: {original_name}.")
    while not (question := input("What would you like to change the name to? ")):
        print("Please enter a valid name.")

    data.attrs["name"] = question
    data["raw_data"].attrs["name"] = question
