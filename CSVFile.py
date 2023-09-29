#
# A class for csv files.
#
# This class does not have much state. It is mostly a convenient way to organize operations on csv files.
#

import os

import numpy as np
import pandas as pd

import csv

import tempfile


class CSVFile:
    def __init__(self, csv_file, dataset_name=None):
        """Initialize from a csv file on disk (the default way to create this object)"""
        self.dataset_name = (
            dataset_name
            if dataset_name is not None
            else os.path.splitext(os.path.basename(csv_file))[0]
        )
        self.csv_file = csv_file

    #################################################################
    # basic functions
    #################################################################
    def get_dataset_name(self):
        """Returns the name of the dataset"""
        return self.dataset_name

    def get_delimiter(self):
        """Returns the delimiter of a csv file"""
        sniffer = csv.Sniffer()
        with open(self.csv_file) as fp:
            delimiter = sniffer.sniff(fp.read(5000)).delimiter
        return delimiter

    def get_feature_names(self):
        """Returns the names of the features in a csv file (a list of strings)"""
        df = self.load_df()
        return df.columns.tolist()

    def load_df(self, header=True, delimiter="auto"):
        """Load a csv file as a pandas data frame."""
        # auto detect the delimiter from the csv file
        if delimiter == "auto":
            delimiter = self.get_delimiter()
        # load the csv file
        df = pd.read_csv(self.csv_file, delimiter=delimiter)
        # optionally, remove the header
        if not header:
            df = df.iloc[1:]
        return df

    def load_rows(self, header=True):
        """Load a csv file as a list of strings, with one string per row."""
        with open(self.csv_file, "r") as f:
            data = f.readlines()
        # remove all trailing newlines
        data = [line.rstrip("\n") for line in data]
        # optionally, remove the header
        if not header:
            data = data[1:]
        return data

    def load_string(self, header=True):
        """Load a csv file as a single string."""
        # load the csv file into a single string
        with open(self.csv_file, "r") as f:
            data = f.read()
        # remove header TODO, this currently only works if header does not contain "\n"
        if not header:
            data = data.split("\n")[1:]
            data = "\n".join(data)
        return data

    def load_array(self, add_feature_names=False):
        """Load a csv file as a 2d numpy array where each entry is a string.

        If add_featrue_names is true, then all entries will have the format "feature_name = feature_value"
        """
        # load csv as a pandas dataframe
        df = self.load_df()
        feature_names = self.get_feature_names()
        # convert all the entries to strings
        df = df.astype(str)
        # strip whitespaces at beginning and end
        df = df.applymap(lambda x: x.strip())
        # if add_feature_names is true, then convert each entry to the format "feature_name = feature_value"
        if add_feature_names:
            for feature_name in feature_names:
                df[feature_name] = feature_name + " = " + df[feature_name]
        # the underlying numpy array
        data = df.values
        return data

    #################################################################
    # more advanced functions
    # we directly construct data for prompts
    #################################################################
    def load_samples(self, add_feature_names=True):
        """
        Returns: description, samples where description is a string and samples is a list of strings.

        Description:
        =======
        Dataset: adult
        Feature Names: Age, WorkClass, fnlwgt, Education, EducationNum, MaritalStatus, Occupation, Relationship, Race, Gender, CapitalGain, CapitalLoss, HoursPerWeek, NativeCountry, Income

        Samples:
        ========
        ['Age = 39, , Income = <=50K', ..., 'Age = 54, , Income = >50K']
        """
        # load the relevant information from the csv file
        dataset_name = self.get_dataset_name()
        feature_names = self.get_feature_names()
        X = self.load_array(add_feature_names=add_feature_names)
        description = f"Dataset: {dataset_name}\nFeature Names: " + ", ".join(
            feature_names
        )
        samples = [", ".join(x) for x in X]
        return description, samples

    def load_cond_samples(
        self,
        cond_feature_names,
        target=None,
        add_description=True,
        add_feature_names=True,
    ):
        """Returns: prefixes, suffixes

        Prefixes:
        =======
        ['Dataset: adult
        Feature Names: Age, WorkClass, fnlwgt, Education, EducationNum, MaritalStatus, Occupation, Relationship, Race, Gender, CapitalGain, CapitalLoss, HoursPerWeek, NativeCountry, Income
        Feature Values: Age = 39, WorkClass = <=50K', ...,
        Target: Income]

        Samples:
        ========
        ['fnlwgt = 1231, .., Income = <=50K', ...]
        """
        # load the relevant information from the csv file
        dataset_name = self.get_dataset_name()
        feature_names = self.get_feature_names()
        # assert that all cond feature name are valid
        assert all(
            [
                cond_feature_name in feature_names
                for cond_feature_name in cond_feature_names
            ]
        ), "Invalid conditional feature names."
        # assert that the target is valid
        if target is not None:
            assert target in feature_names, "Invalid target."
        X = self.load_array(add_feature_names=add_feature_names)
        cond_feature_indices = [
            feature_names.index(name) for name in cond_feature_names
        ]
        sample_feature_indices = [
            idx for idx in range(len(feature_names)) if idx not in cond_feature_indices
        ]
        description = f"Dataset: {dataset_name}\nFeature Names: " + ", ".join(
            feature_names
        )
        # prefixes include the cond_feature_names
        prefixes = [", ".join(x) for x in X[:, cond_feature_indices]]
        if add_description:
            prefixes = [
                description + "\nFeature Values: " + prefix for prefix in prefixes
            ]
        if target is not None:
            prefixes = [prefix + "\nTarget: " + target for prefix in prefixes]
        if target is None:
            suffixes = [", ".join(x) for x in X[:, sample_feature_indices]]
        else:
            target_index = feature_names.index(target)
            suffixes = [x[target_index] for x in X]
        return prefixes, suffixes

    @classmethod
    def from_df(cls, df, dataset_name):
        """Initialize a CSVFile object from a pandas dataframe."""
        # get the path on disk of this python file
        # path = os.path.dirname(os.path.abspath(__file__))
        # create a 'tmp' folder if it does not already exist
        # tmp_folder = os.path.join(path, "tmp")
        # if not os.path.exists(tmp_folder):
        #    os.makedirs(tmp_folder)
        # create a temporary folder for our dataset
        tmp_folder = tempfile.mkdtemp()
        # we save the pandas dataframe in the temporary folder, using the name of the dataset
        filename = os.path.join(tmp_folder, f"{dataset_name}.csv")
        df.to_csv(filename, index=False)
        # initialize a new CSVFile object from this file
        return cls(filename)


# simple tests
if __name__ == "__main__":
    csv_file = CSVFile("csv/california-housing.csv")
    print(csv_file.get_dataset_name())
    print(csv_file.get_feature_names())
    print(csv_file.load_rows()[1])

    df = csv_file.load_df()
    # randomly permute the columns
    df = df.sample(frac=1, axis=1)
    csv_file = CSVFile.from_df(df, "California Housing")
    print(csv_file.get_dataset_name())
    print(csv_file.get_feature_names())
    print(csv_file.load_rows()[1])
