####################################################################################
# This file contains different chat completion tasks.
#
# The functions in this file generate, format and send prompts, based on
# the provided csv files. They return the raw model responses, and do not
# perform any tests or analysis. Different tests make use
# of the same chat completion functions.
#
# Almost all test and completions are based on prefix_suffix_chat_completion.
####################################################################################


import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import yaml

import experiment_utils

import openai_api


import csv

import analysis

import utils
from utils import load_csv_rows, get_delimiter, find_nth

from row_independence import statistical_feature_prediction_test

from CSVFile import CSVFile


####################################################################################
# Feature values chat completion function. This function is used for sampling,
# conditional sampling, and prediction.
####################################################################################


def feature_values_chat_completion(
    csv_file,
    system_prompt,
    num_queries,
    few_shot=[],  # list or integer
    cond_feature_names=[],
    fs_cond_feature_names=[],  # a list of lists of conditional feature names for each few-shot example
    add_description=True,
    out_file=None,
):
    """Feature chat completion task. This task asks the LLM to complete the feature values of observations in the dataset.

    The prompt format is the following:
        System: <system_prompt>
            |
            | {few_shot} examples from other csv files.
            |
        User: Dataset: <dataset_name>
              Feature Names: Feature 1, Feature 2, ..., Feature n
              Feature Values: Feature 1 = value 1, Feature 2 = value 2, ..., Feature m = value m
              [Target: Feature k]
        Response: Feature m + 1 = value m + 1, ..., Feature n = value n [Feature k = value k]

    This can be modified in the following ways:
        - Remove dataset description and feature names ({add_description} parameter)
        - don't provide any conditional features
        - Don't use the feature names, but only the values.   (TODO ? or maybe remove, latter for formatter class)

    Options:
        - few_shot: use few-shot examples from other csv files (list), or few_shot examples from the same csv file (int)
        - target & fs_targets: if target is not None, then the LLM is asked to complete only the value of the target feature.

    The feature names are ordered in the prompt as they are ordered in the csv file. In the future we might want to relax this.
    """
    # TODO assert that all the given feature names are valid (i.e. occur in the dataset, otherwise throw exception)

    # csv file strings to csv file objects
    if isinstance(csv_file, str):
        csv_file = CSVFile(csv_file)
    if isinstance(few_shot, list):
        for idx in range(len(few_shot)):
            if isinstance(few_shot[idx], str):
                few_shot[idx] = CSVFile(few_shot[idx])

    dataset_name = csv_file.get_dataset_name()
    conditional_sampling = (
        cond_feature_names is not None and len(cond_feature_names) > 0
    )

    # if the few-shot argument is a list, then csv_file should not be in there
    # the current option is to remove it (TODO issue warning)
    if isinstance(few_shot, list):
        few_shot = [x for x in few_shot if not dataset_name in x.get_dataset_name()]

    # if few-shot is an integer, then include few_shot examples from csv_file
    # this is implemented by replacing few_shot and fs_cond_feature_names with the appropriate lists
    if isinstance(few_shot, int):
        few_shot = [csv_file for _ in range(few_shot)]
        fs_cond_feature_names = [cond_feature_names for _ in range(len(few_shot))]

    # issue a warning if conditional_sampling, but no fs_cond_feature_names
    if conditional_sampling and len(few_shot) > 0 and len(fs_cond_feature_names) == 0:
        print(
            experiment_utils.bcolors.WARNING
            + "WARNING: feature_chat_completion: Conditional sampling, but no conditional feature names for the few-shot examples provided."
            + experiment_utils.bcolors.ENDC
        )

    # prefixes and suffixes for the main dataset
    if conditional_sampling:
        prefixes, samples = csv_file.load_cond_samples(
            cond_feature_names, add_description=add_description
        )
    else:
        prefix, samples = csv_file.load_samples()
        prefixes = [prefix] * len(samples)

    # prefixes and suffixes for the few-shot examples
    few_shot_prefixes_suffixes = []
    for fs_idx, fs_csv_file in enumerate(few_shot):
        if conditional_sampling:
            fs_prefixes, fs_samples = fs_csv_file.load_cond_samples(
                fs_cond_feature_names[fs_idx],
                add_description=add_description,
            )
            few_shot_prefixes_suffixes.append((fs_prefixes, fs_samples))
        else:
            fs_prefix, fs_samples = fs_csv_file.load_samples()
            few_shot_prefixes_suffixes.append(
                ([fs_prefix] * len(fs_samples), fs_samples)
            )

    # execute chat queries
    test_prefixes, test_suffixes, responses = prefix_suffix_chat_completion(
        prefixes,
        samples,
        system_prompt,
        few_shot=few_shot_prefixes_suffixes,
        num_queries=num_queries,
        out_file=out_file,
    )

    return test_prefixes, test_suffixes, responses


####################################################################################
# The row chat completion task. This task ask the LLM to predict the next row in the
# csv file, given the previous rows. This task is the basis for the row completion
# test, and also for the first token test.
####################################################################################


def row_chat_completion(
    csv_file,
    system_prompt,
    num_prefix_rows=10,
    num_queries=100,
    few_shot=7,
    out_file=None,
):
    """Row  chat completion task. This task ask the LLM to predict the next row in the
    csv file, given the previous rows. This task is the basis for the row completion
    test, and also for the first token test. Uses prefix_suffix_chat_completion."""
    # assert that few_shot is an integer
    assert isinstance(few_shot, int), "For row completion, few_shot must be an integer."

    # load the file as a list of strings
    data = load_csv_rows(csv_file)

    # prepare data
    prefixes = []
    suffixes = []
    for idx in range(len(data) - num_prefix_rows):
        prefixes.append("\n".join(data[idx : idx + num_prefix_rows]))
        suffixes.append(data[idx + num_prefix_rows])

    test_prefixes, test_suffixes, responses = prefix_suffix_chat_completion(
        prefixes,
        suffixes,
        system_prompt,
        few_shot=few_shot,
        num_queries=num_queries,
        out_file=out_file,
    )

    return test_prefixes, test_suffixes, responses


####################################################################################
# Almost all of the different tests that we perform
# can be cast in the prompt structue of
# 'prefix-suffix chat completion'.
# This is implemented by the following function.
####################################################################################


def prefix_suffix_chat_completion(
    prefixes: list,
    suffixes: list,
    system_prompt: str,
    few_shot=None,
    num_queries=100,
    out_file=None,
    rng=None,
):
    """A basic chat completion function. Takes a list of prefixes and suffixes and a system prompt.
    Sends {num_queries} prompts of the format

    System: <system_prompt>
        User: <prefix>          |
        Assistant: <suffix>     |
        ...                     | {few_shot} times, or one example from each (prefixes, suffixes) pair in a {few_shot} list.
        User: <prefix>          | In the second case, few_shot = [([prefixes], [suffixes]), ..., ([prefixes], [suffixes])]
        Assistant: <suffix>     |
    User: <prefix>
    Assistant: <response> (=  test suffix?)

    The num_queries prefixes and suffixes are randomly selected from the respective lists.
    The function guarantees that the test suffix (as a complete string) is not contained in any of the few-shot prefixes or suffixes.

    Stores the results in a csv file.

    Returns: the test prefixes, test suffixes, and responses
    """
    assert len(prefixes) == len(
        suffixes
    ), "prefixes and suffixes must have the same length"

    # randomly shuffle the prefixes and suffixes
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.permutation(len(prefixes))
    prefixes = [prefixes[i] for i in idx]
    suffixes = [suffixes[i] for i in idx]

    # the number of points to evaluate
    num_points = min(num_queries, len(prefixes))

    test_prefixes = []
    test_suffixes = []
    responses = []
    for i_testpoint in range(num_points):
        # system prompt
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]
        # few-shot examples?
        if few_shot is not None:
            # if few_shot is an integer, include few_shot examples from the original prefixes and suffixes
            if isinstance(few_shot, int):
                for _ in range(few_shot):
                    idx = None
                    # select a random prefix/suffix pair
                    while (
                        idx is None
                        or idx == i_testpoint
                        # assert that the test suffix is not contained in the few-shot prefixes or suffixes
                        or suffixes[i_testpoint] in prefixes[idx]
                        or suffixes[i_testpoint] in suffixes[idx]
                    ):
                        idx = rng.choice(len(prefixes))
                    prefix = prefixes[idx]
                    suffix = suffixes[idx]
                    messages.append({"role": "user", "content": prefix})
                    messages.append({"role": "assistant", "content": suffix})
            # if few_shot is a list of (prefixes, suffixes)-tuples, inlude one example from each tuple
            elif isinstance(few_shot, list):
                for fs_prefixes, fs_suffixes in few_shot:
                    fs_prefix, fs_suffix = None, None
                    # select a random prefix/suffix pair
                    while (
                        fs_prefix is None
                        # assert that the test suffix is not contained in the few-shot prefixes or suffixes
                        or suffixes[i_testpoint] in fs_prefix
                        or suffixes[i_testpoint] in fs_suffix
                    ):
                        fs_idx = rng.choice(len(fs_prefixes))
                        fs_prefix = fs_prefixes[fs_idx]
                        fs_suffix = fs_suffixes[fs_idx]
                    messages.append({"role": "user", "content": fs_prefix})
                    messages.append({"role": "assistant", "content": fs_suffix})

        # test observation
        test_prefix = prefixes[i_testpoint]
        test_suffix = suffixes[i_testpoint]
        messages.append({"role": "user", "content": test_prefix})
        # send message
        try:
            response = experiment_utils.send_chat_completion(messages)
            responses.append(response["choices"][0]["message"]["content"])
        except Exception as e:
            print(e)  # TODO repeat the entry in the foor loop with another query
            responses.append(
                ""
            )  # equate a failure to send the query with an empty model response
        test_prefixes.append(test_prefix)
        test_suffixes.append(test_suffix)

    # save the results to file
    if out_file is not None:
        results_df = pd.DataFrame(
            {
                "prefix": test_prefixes,
                "suffix": test_suffixes,
                "response": responses,
            }
        )
        results_df.to_csv(
            out_file,
            index=False,
        )

    return test_prefixes, test_suffixes, responses
