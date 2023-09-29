####################################################################################
# This file contains the higher-level functions that implement the main
# logic of csvllm. The functions in this file implement the different kinds
# of tasks that can be performed by the tool. The functions in this file
# are a little more general, which means that some of the functions implement
# more than one task.
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

from chat_completions import (
    prefix_suffix_chat_completion,
    row_chat_completion,
    feature_values_chat_completion,
)

from CSVFile import CSVFile

####################################################################################
# Tests for what the model knows about the data
####################################################################################


# TODO MAKE IT ZERO SHOT FOR THE CASE THAT FEW_SHOT_CSV_FILES IS NONE
def csv_format_test(
    csv_file,
    system_prompt,
    num_points=5,
    few_shot_csv_files=[
        "csv/IRIS.csv",
        "csv/uci-wine.csv",
        "csv/titanic-train.csv",
    ],
    verbose=True,
):
    """Does the model know the format of the csv file?"""

    # the name of the file without the path and extension
    dataset_name = os.path.basename(csv_file).split(".")[0]

    # load all the different csv files
    data = load_csv_rows(csv_file, header=False)
    few_shot_data = [load_csv_rows(f, header=False) for f in few_shot_csv_files]

    # run th experiment
    exp_name = f"{dataset_name}-format-test"
    experiment_utils.set_logging_task(exp_name)

    test_suffixes = []
    for i_testpoint in range(num_points):
        # system prompt
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]
        for few_shot_idx in range(len(few_shot_data)):
            # select a random datapoint
            idx = np.random.randint(0, len(few_shot_data[few_shot_idx]))
            # the datapoint
            data_point = few_shot_data[few_shot_idx][idx]
            # split the string in prefix and suffix of equal size
            prefix = data_point[: len(data_point) // 2]
            suffix = data_point[len(data_point) // 2 :]
            # add the prefix and suffix to the messages
            messages.append({"role": "user", "content": prefix})
            messages.append({"role": "assistant", "content": suffix})
        # select a random test point
        idx = np.random.randint(0, len(data))
        # the test point
        data_point = data[idx]
        # split the string in prefix and suffix of equal size
        prefix = data_point[: len(data_point) // 2]
        suffix = data_point[len(data_point) // 2 :]
        # add the prefix to the messages
        messages.append({"role": "user", "content": prefix})
        test_suffixes.append(suffix)
        # send message
        try:
            response = experiment_utils.send_chat_completion(
                messages,
                max_tokens=100,
            )
            completion = response["choices"][0]["message"]["content"]
            print(">>> BEGIN ROW ")
            print("PREFIX: ", prefix)
            print("LLM COMPLETION: ", completion)
            print("TRUE COMPLETION: ", suffix)
            print(">>> END ROW ")
        # catch and print exceptions
        except Exception as e:
            print(e)  # TODO repeat the entry in the foor loop with another query


def feature_names_test(
    csv_file, system_prompt, num_prefix_features, few_shot_csv_files=[], save_path=None
):
    """Test if the model knows the names of the features.

    The prompt format is:
        System: <system_prompt>
        User: Dataset: <dataset_name>
              Feature 1, Feature 2, ..., Feature n
        Response: Feature n+1, Feature n+2, ..., Feature m

    This can be modified in the following ways:
    - Include few-shot examples from other csv files.
    """
    dataset_name = utils.get_dataset_name(csv_file)
    feature_names = utils.get_feature_names(csv_file)

    # remove the current csv file from the few-shot csv files should it be present there
    few_shot_csv_files = [x for x in few_shot_csv_files if not dataset_name in x]

    # setup for the few-shot examples
    fs_dataset_names = [utils.get_dataset_name(x) for x in few_shot_csv_files]
    fs_feature_names = [
        utils.get_feature_names(fs_csv_file) for fs_csv_file in few_shot_csv_files
    ]
    fs_prefix_feature = [
        utils.adjust_num_prefix_features(csv_file, num_prefix_features, fs_csv_file)
        for fs_csv_file in few_shot_csv_files
    ]

    # construt the prompt
    prefixes = [
        f"Dataset: {dataset_name}. Feature Names: "
        + ", ".join(feature_names[:num_prefix_features])
    ]
    suffixes = [", ".join(feature_names[num_prefix_features:])]

    few_shot = []
    for fs_dataset_name, fs_feature_name, fs_prefix_feature in zip(
        fs_dataset_names, fs_feature_names, fs_prefix_feature
    ):
        few_shot.append(
            (
                [
                    f"Dataset: {fs_dataset_name}. Feature Names: "
                    + ", ".join(fs_feature_name[:fs_prefix_feature])
                ],
                [", ".join(fs_feature_name[fs_prefix_feature:])],
            )
        )

    # send the prompt
    exp_name = dataset_name + "-feature-names"
    experiment_utils.set_logging_task(exp_name)

    test_prefixes, test_suffixes, responses = prefix_suffix_chat_completion(
        prefixes,
        suffixes,
        system_prompt,
        few_shot=few_shot,
        num_queries=1,
    )

    # TODO do some sort of evaluation
    print(responses)
    print(test_suffixes)


####################################################################################
# Test for memorization
####################################################################################


def header_test(csv_file, system_prompt, few_shot_csv_files=[], completion_length=500):
    """Header test with a chat model, using other csv files as few-shot examples.

    The test consists of 4 queries that split the csv file at random positions in rows 2, 4, 6, and 8. The test then reports the best completion.

    NOTE: This test might fail if the header and rows of the csv file are very long, and the model has a small context window.
    NOTE: in the end, this is the case for all of our tests :)
    """
    # load the csv file as a single contiguous string. also load the rows to determine offsets within the string
    data = utils.load_csv_string(csv_file, header=True)
    csv_rows = utils.load_csv_rows(csv_file, header=True)

    # load the few-shot examples
    few_shot_data = []
    for fs_csv_file in few_shot_csv_files:
        fs_data = utils.load_csv_string(fs_csv_file, header=True)
        few_shot_data.append(fs_data)

    # perform the test 5 times, cutting the dataset at random positions in rows 2, 4, 6, and 8
    num_completions = -1
    header, completion = None, None
    for i_row in [2, 4, 6, 8]:
        offset = np.sum([len(row) for row in csv_rows[: i_row - 1]])
        offset += np.random.randint(
            len(csv_rows[i_row]) // 3, 2 * len(csv_rows[i_row]) // 3
        )
        prefixes = [data[:offset]]
        suffixes = [data[offset : offset + completion_length]]
        few_shot = [
            ([fs_data[:offset]], [fs_data[offset : offset + completion_length]])
            for fs_data in few_shot_data
        ]

        _, _, response = prefix_suffix_chat_completion(
            prefixes, suffixes, system_prompt, few_shot=few_shot, num_queries=1
        )

        # find the first digit where the response and the completion disagree
        idx = -1000
        for idx, (c, r) in enumerate(zip(data[offset:], response[0])):
            if c != r:
                break
        if idx == len(response[0]) - 1 and response[0][idx] == data[offset + idx]:
            idx += 1  # no disagreement found, set idx to length of the response

        # is this the best completion so far?
        if idx > num_completions:
            num_completions = idx
            header = prefixes[0]
            completion = response[0]

    # for the printing, we first color all green up to the first disagreement
    completion_print = experiment_utils.bcolors.Green + completion[:num_completions]

    # then color red up to the beginning of the next row, if any
    remaining_completion = completion[num_completions:]
    idx = remaining_completion.find("\n")
    if idx == -1:
        completion_print += experiment_utils.bcolors.Red + remaining_completion
    else:
        completion_print += (
            experiment_utils.bcolors.Red + remaining_completion[:idx] + "\n"
        )
        remaining_completion = remaining_completion[idx + 1 :]

        # for all additional rows, green up to the first disagreement, all red after that
        completion_rows = remaining_completion.split("\n")

        # the corresponding next row in the csv file
        data_idx = data[len(header) + num_completions :].find("\n")
        data_rows = data[len(header) + num_completions + data_idx + 1 :].split("\n")

        for completion_row, data_row in zip(completion_rows, data_rows):
            if completion_row == data_row:
                completion_print += (
                    experiment_utils.bcolors.Green + completion_row + "\n"
                )
                continue
            # not equal, find the first disagreement
            idx = -1000
            for idx, (c, r) in enumerate(zip(data_row, completion_row)):
                if c != r:
                    break
            if idx == len(completion_row) - 1 and completion_row[idx] == data_row[idx]:
                idx += 1
            # print first part green, second part red
            completion_print += (
                experiment_utils.bcolors.Green
                + completion_row[:idx]
                + experiment_utils.bcolors.Red
                + completion_row[idx:]
                + "\n"
            )

    # print the result
    print(
        experiment_utils.bcolors.BOLD
        + "Header Test: "
        + experiment_utils.bcolors.ENDC
        + experiment_utils.bcolors.Black
        + header
        + completion_print
        + experiment_utils.bcolors.ENDC
    )


def row_completion_test(
    csv_file,
    system_prompt,
    num_prefix_rows=10,
    num_queries=100,
    few_shot=7,
    out_file=None,
):
    """Row completion test: Complete the next row of the csv file, given the previous rows."""

    # log model queries
    dataset_name = utils.get_dataset_name(csv_file)
    exp_name = f"{dataset_name}-row-completion"
    experiment_utils.set_logging_task(exp_name)

    # ask the model to perform row chat completion
    test_prefixes, test_suffixes, responses = row_chat_completion(
        csv_file, system_prompt, num_prefix_rows, num_queries, few_shot, out_file
    )

    # count the number of exact matches
    # NOTE here we assume that the test suffix is a single row that is unique, i.e. no duplicate rows
    num_exact_matches = 0
    for test_suffix, response in zip(test_suffixes, responses):
        if test_suffix.strip() in response.strip():
            num_exact_matches += 1

    # the statistical test using the levenshtein distance
    test_prefix_rows = [prefix.split("\n") for prefix in test_prefixes]
    test_result = analysis.levenshtein_distance_t_test(
        responses, test_suffixes, test_prefix_rows
    )

    # print the result
    print(
        experiment_utils.bcolors.BOLD
        + "Row Completion Test: "
        + experiment_utils.bcolors.ENDC
        + experiment_utils.bcolors.Black
        + f"{num_exact_matches}/{num_queries} exact matches. Levenshtein distance test p-value: {test_result.pvalue:.3f}."
        + experiment_utils.bcolors.ENDC
    )

    return test_prefixes, test_suffixes, responses


def first_token_test(
    csv_file,
    system_prompt,
    num_prefix_rows=10,
    num_queries=100,
    few_shot=7,
    save_path=None,
):
    # first, determine the number of digits that the first token should have
    num_digits = analysis.build_first_token(csv_file)

    # then, run a feature prediction test to see if the first token is actually random
    df = utils.load_csv_df(csv_file)
    rows = utils.load_csv_rows(csv_file, header=False)
    df["FIRST_TOKEN_TEST_ROW"] = [r[:num_digits] for r in rows]
    df["FIRST_TOKEN_TEST_ROW"] = df["FIRST_TOKEN_TEST_ROW"].astype(str)
    # save the df to a tmp csv file (TODO allow data frames as input to the tests?)
    tmp_csv_file = os.path.join(save_path, "tmp.csv")
    df.to_csv(tmp_csv_file, index=False)
    rejected = statistical_feature_prediction_test(
        tmp_csv_file,
        "FIRST_TOKEN_TEST_ROW",
        num_prefix_rows=5,
        confidence_level=0.99,
    )

    # the most common first token
    most_common_first_token = df["FIRST_TOKEN_TEST_ROW"].value_counts().index[0]
    print(most_common_first_token)

    # if the feature prediction test rejects randomness, refuse to run the test
    if rejected:
        print(
            experiment_utils.bcolors.FAIL
            + "Aborting the test because the first token does not seem to be random. The most likely reason for this is that the rows in the csv file are not random. For example, the first feature might be the id of the observation."
            + experiment_utils.bcolors.ENDC
        )
        return

    #  set max_tokens to the number of digits (speedup)
    experiment_utils.llm_max_tokens = num_digits

    # perform a row completion task
    test_prefixes, test_suffixes, responses = row_chat_completion(
        csv_file, system_prompt, num_prefix_rows, num_queries, few_shot, save_path
    )

    # parse responses
    test_tokens = [x[:num_digits] for x in test_suffixes]
    response_tokens = [x[:num_digits] for x in responses]

    # count number of exact matches
    num_exact_matches = np.sum(np.array(test_tokens) == np.array(response_tokens))

    # count the number of exact matches using the most common first token
    num_exact_matches_most_common = np.sum(
        np.array(response_tokens) == most_common_first_token
    )

    # print result
    print(
        experiment_utils.bcolors.BOLD
        + "First Token Test: "
        + experiment_utils.bcolors.ENDC
        + experiment_utils.bcolors.Black
        + f"{num_exact_matches}/{num_queries} exact matches. Most common first token: {num_exact_matches_most_common}/{num_queries}."
        + experiment_utils.bcolors.ENDC
    )


def feature_completion_test(
    csv_file,
    system_prompt,
    feature_name: str,
    num_queries=100,
    few_shot=5,
    out_file=None,
):
    """Feature completion test where we attempt to predict a single rare feature & count the number of exact matches.

    The basic prompt format is the following:
        System: <system_prompt>
        User: Feature 1 = value 1, Feature 2 = value 2, ..., Feature n = value n
        Response: Feature {feature_name} = value

    This can be modified in the following ways:
        - Include few-shot examples from other csv files.
        - Don't use the feature names, but only the values.
    """
    # TODO statistical analysis of the uniqueness of the feature (i.e., is the test appropriate?!)

    # if no feature value is provided, automatically select the most unique feature
    if feature_name is None:
        feature_name, frac_unique_values = analysis.find_most_unique_feature(csv_file)
        print(
            experiment_utils.bcolors.BOLD
            + "Info: "
            + experiment_utils.bcolors.ENDC
            + f"Using feature {feature_name} with {100*frac_unique_values:.2f}% unique values."
        )

    # all the other features are the conditional features
    feature_names = utils.get_feature_names(csv_file)
    cond_feature_names = [f for f in feature_names if f != feature_name]

    # log model queries
    dataset_name = utils.get_dataset_name(csv_file)
    exp_name = f"{dataset_name}-feature-completion"
    experiment_utils.set_logging_task(exp_name)

    # run the test
    test_prefixes, test_suffixes, responses = feature_values_chat_completion(
        csv_file,
        system_prompt,
        num_queries,
        few_shot,
        cond_feature_names,
        add_description=False,
        out_file=out_file,
    )

    # parse the model responses
    response_df = utils.parse_feature_stings(responses, [feature_name])
    test_suffix_df = utils.parse_feature_stings(test_suffixes, [feature_name])

    # count number of exact matches
    num_exact_matches = np.sum(
        response_df[feature_name] == test_suffix_df[feature_name]
    )

    # print the result
    print(
        experiment_utils.bcolors.BOLD
        + f"Feature Completion Test (Feature {feature_name}): "
        + experiment_utils.bcolors.ENDC
        + experiment_utils.bcolors.Black
        + f"{num_exact_matches}/{num_queries} exact matches."
        + experiment_utils.bcolors.ENDC
    )


####################################################################################
# Prediction and sampling
####################################################################################
from collections import Counter


def predict(csv_file, system_prompt, feature_name: str, num_queries=100, few_shot=7):
    """Predict the feature {feature_name}, using all the other features in the csv file.

    Reports the accuracy / mse.

    The basic prompt format is the following:
        System: <system_prompt>
        User: Feature 1 = value 1, Feature 2 = value 2, ..., Feature n = value n
        Response: Target Feature = value

    Inludes {few_shot} examples from the same csv file.

    TODO allow to stratify the few-shot examples by the target feature.
    """
    # all the other features are the conditional features
    feature_names = utils.get_feature_names(csv_file)
    cond_feature_names = [f for f in feature_names if f != feature_name]

    # run the test
    test_prefixes, test_suffixes, responses = feature_values_chat_completion(
        csv_file,
        system_prompt,
        num_queries,
        few_shot,
        cond_feature_names,
        add_description=False,
    )

    # parse the model responses
    response_df = utils.parse_feature_stings(responses, [feature_name])
    test_suffix_df = utils.parse_feature_stings(test_suffixes, [feature_name])

    # is the classification or regression?
    df = utils.load_csv_df(csv_file)
    is_classification = False
    if df[feature_name].dtype == "object":
        is_classification = True

    # compute the accuracy/mse
    y_true = test_suffix_df[feature_name]
    y_pred = response_df[feature_name]

    if is_classification:
        score, ci = utils.accuracy(y_true, y_pred)
        print(
            experiment_utils.bcolors.BOLD
            + "Accuracy: "
            + experiment_utils.bcolors.ENDC
            + f"{score:.3} ({ci.low:.3}, {ci.high:.3})"
        )
        # TODO replace test with train here
        baseline_score, baseline_ci = utils.accuracy(
            y_true, np.repeat(Counter(y_true).most_common(1)[0][0], len(y_true))
        )
        print(
            experiment_utils.bcolors.BOLD
            + "Baseline (most common class): "
            + experiment_utils.bcolors.ENDC
            + f"{baseline_score:.3} ({baseline_ci.low:.3}, {baseline_ci.high:.3})"
        )
    else:
        y_true = y_true.astype(float)
        y_pred = y_pred.astype(float)
        score, ci = utils.mse(y_true, y_pred)
        print(
            experiment_utils.bcolors.BOLD
            + "Mean-squared-error: "
            + experiment_utils.bcolors.ENDC
            + f"{score:.3} ({ci.low:.3}, {ci.high:.3})"
        )
        baseline_score, baseline_ci = utils.mse(
            y_true, np.repeat(np.mean(y_true), len(y_true))
        )
        print(
            experiment_utils.bcolors.BOLD
            + "Baseline (mean): "
            + experiment_utils.bcolors.ENDC
            + f"{baseline_score:.3} ({baseline_ci.low:.3}, {baseline_ci.high:.3})"
        )


def sample(
    csv_file,
    system_prompt,
    num_queries,
    few_shot=[],
    cond_feature_names=[],
    out_file=None,
):
    """zero-shot sampling from the csv file, using few-shot examples from other csv files."""
    # few_shot has to be a list
    if not isinstance(few_shot, list):
        raise ValueError("For sampling, few_shot has to be a list")

    # log model queries
    dataset_name = utils.get_dataset_name(csv_file)
    task_name = f"{dataset_name}-sample"
    experiment_utils.set_logging_task(task_name)

    # run the test
    test_prefixes, test_suffixes, responses = feature_values_chat_completion(
        csv_file,
        system_prompt,
        num_queries,
        few_shot,
        cond_feature_names,
        add_description=True,
        out_file=None,
    )

    # we use the descriptive
    if len(cond_feature_names) > 0:
        pass
        # TODO handle the condtional case!

    # parse the model output. we are interested in the final samples
    feature_names = utils.get_feature_names(csv_file)
    response_df = utils.parse_feature_stings(responses, feature_names)

    # save the dataframe with the final samples
    if out_file is not None:
        print(out_file)
        response_df.to_csv(out_file, index=False)


####################################################################################
# Mode test and conditional completion test for learning (does the model know the most
# important statistics of the data distribution?)
####################################################################################


# TODO
#def mode_test():
#    pass


def conditional_completion_test(
    csv_file: str,
    system_prompt,
    feature_name: str,
    num_queries=250,
    prefix_length=[0, 0.25, 0.5, 0.75, 1],
    few_shot=[],
    out_file=None,
):
    """Conditional completion test for conditional distribution modelling.

    The task is to always predict the feature {feature_name}, give different conditional features.

    The prompt format is the following:
        System: <system_prompt>
            |
            | {few_shot} examples from other csv files.
            |
        User: Dataset: <dataset_name>
              Feature Names: Feature 1, Feature 2, ..., Feature m, Feature {feature_name}, Feature m + 2, ..., Feature n
              Feature Values: Feature 1 = value 1, Feature 2 = value 2, ..., Feature m = value m
        Response: {feature_name} = value m + 1 [, Feature {feature_name}, Feature m + 2, ..., Feature n]

    We ask the model to provide {num_queries} completions for each prefix length in {prefix_length}.

    The ordering of the feature names will be the same as in the csv file, except for the feature {feature_name}.

    The test computes the p-value of the hypothesis that the completions are unconditional.
    """
    # all the other features are the conditional features
    feature_names = utils.get_feature_names(csv_file)
    cond_feature_names = [f for f in feature_names if f != feature_name]

    # in this task, the order of the features is not the same as in the original csv file.
    # therefore, we shuffle the order of the features in the few-shot csv files.
    shuffled_few_shot = []
    for fs_csv_file in few_shot:
        df = utils.load_csv_df(fs_csv_file)
        df = df.sample(frac=1, axis=1)
        shuffled_few_shot.append(
            CSVFile.from_df(df, utils.get_dataset_name(fs_csv_file)).csv_file
        )
    few_shot = shuffled_few_shot

    # feature names in the few-shot files
    fs_fns = [utils.get_feature_names(fs_csv_file) for fs_csv_file in few_shot]

    # fs_target_fns = [np.random.choice(fns) for fns in fs_fns]
    # fs_cond_fns = [
    #    [f for f in fns if f != fs_target_fn]
    #    for fns, fs_target_fn in zip(fs_fns, fs_target_fns)
    # ]

    # estimate and set the maximum number of tokens to speed up the experiment
    df = utils.load_csv_df(csv_file)
    max_tokens = (
        df[feature_name]
        .astype(str)
        .apply(
            lambda x: experiment_utils.num_tokens_from_string(f"{feature_name} = " + x)
        )
        .astype(int)
        .max()
    )
    experiment_utils.llm_max_tokens = int(1.1 * max_tokens) + 3

    # create a data frame to hold the model responses
    df = pd.DataFrame(columns=["num_prefix_features"].extend(feature_names))
    # one completion task for each prefix length
    for p in prefix_length:
        # the conditional features that we use
        p_cond_fns = cond_feature_names[: int(p * len(cond_feature_names))]
        p_fs_cond_fns = [fns[: min(int(p * len(fns)), len(fns) - 1)] for fns in fs_fns]

        # we create a temporary version of the dataset where the features are ordered as {conditional features}, {target feature}, {other features}
        # this is the order in which we will ask the model to predict the target feature
        p_df = utils.load_csv_df(csv_file)
        p_df = p_df[
            [
                *p_cond_fns,
                feature_name,
                *[f for f in cond_feature_names if f not in p_cond_fns],
            ]
        ]
        p_csv_file = CSVFile.from_df(p_df, utils.get_dataset_name(csv_file)).csv_file

        # run the task
        test_prefixes, test_suffixes, responses = feature_values_chat_completion(
            p_csv_file,
            system_prompt,
            num_queries,
            few_shot,
            p_cond_fns,
            p_fs_cond_fns,
            add_description=True,
            out_file=None,
        )

        # parse the prefixes and response
        test_prefixes_df = utils.parse_feature_stings(test_prefixes, feature_names)
        response_df = utils.parse_feature_stings(responses, feature_names)

        # drop all columns other than feature_name from the response_df
        response_df = response_df[[feature_name]]

        # add all the columns in the response_df to the test_suffix_df
        test_prefixes_df = pd.concat([test_prefixes_df, response_df], axis=1)

        # add empty columns for the features that are not in the prefix
        for f in feature_names:
            if f not in test_prefixes_df.columns:
                test_prefixes_df[f] = [""] * test_prefixes_df.shape[0]

        # add the number of prefix features as a column
        test_prefixes_df["num_prefix_features"] = [
            len(p_cond_fns)
        ] * test_prefixes_df.shape[0]

        # store results
        df = pd.concat(
            [
                df,
                test_prefixes_df,
            ],
            ignore_index=True,
        )

    # save the dataframe
    if out_file is not None:
        df.to_csv(out_file, index=False)

    # analysis, given the dataframe with the results
    return analysis.conditional_completion_test(df)


def ordered_completion(
    csv_file,
    system_prompt,
    feature_name: str,
    num_queries=250,
    few_shot=[],
    out_file=None,
):
    """Ordered, conditional completion of feature {feature_name}.

    The prompt format is the following:
        # TODO
    """
    feature_names = utils.get_feature_names(csv_file)
    fs_feature_names = [
        utils.get_feature_names(fs_csv_file) for fs_csv_file in few_shot
    ]

    # the index of the completion feature
    feature_idx = feature_names.index(feature_name)

    # the relative number of feature names that we use as conditional features
    p = feature_idx / len(feature_names)
    p = max(0.001, min(0.999, p))

    # TODO handle edge case no conditional features
    cond_feature_names = utils.get_prefix_features(csv_file, p)
    fs_cond_feature_names = [
        utils.get_prefix_features(fs_csv_file, p) for fs_csv_file in few_shot
    ]

    # estimate and set the maximum number of tokens to speed up the experiment
    df = utils.load_csv_df(csv_file)
    max_tokens = (
        df[feature_name]
        .astype(str)
        .apply(
            lambda x: experiment_utils.num_tokens_from_string(f"{feature_name} = " + x)
        )
        .astype(int)
        .max()
    )
    experiment_utils.llm_max_tokens = int(1.1 * max_tokens) + 3
    print(f"Setting max_tokens to {experiment_utils.llm_max_tokens}")

    # log model queries
    dataset_name = utils.get_dataset_name(csv_file)
    exp_name = f"{dataset_name}-ordered-completion-{feature_name}"
    experiment_utils.set_logging_task(exp_name)

    # run the task
    test_prefixes, test_suffixes, responses = feature_values_chat_completion(
        csv_file,
        system_prompt,
        num_queries,
        few_shot,
        cond_feature_names,
        fs_cond_feature_names,
        add_description=True,
        out_file=out_file,
    )
