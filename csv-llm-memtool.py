import os

import numpy as np
import pandas as pd


import yaml

import experiment_utils

from openai_api import openai_api_setup

import test_functions
import analysis
import utils
import copy

import csv

from row_independence import row_independence_test


if __name__ == "__main__":
    # parse args
    import argparse

    parser = argparse.ArgumentParser(
        prog="csv-llm-memtest (version 0.1.0)",
        description="A python tool for csv files and large language models. Provides different memorization tests. Use --debug to see the prompts.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # required args
    tasks = {
        "all": "Automatically perform a series of memorization tests and report the results.",
        "predict": "Few-shot prediction of a feature in the csv file.",
        "sample": "Ask the LLM to provide random samples from the csv file. For conditional sampling, use the parameter --cond.",
        "feature-names": "Does the LLM to complete the feature names from the top row of the csv file?",
        "feature-values": "Does the LLM produce feature values of the same format as in the csv file?",
        "mode": "Mode test for unconditional distribution modelling.",
        "ordered-completion": "Feature completion, respecting the order of the features in the csv file.",
        "header": "Header test for memorization.",
        "row-completion": "Row completion test for memorization.",
        "feature-completion": "Unique feature completion test for memorization.",
        "first-token": "First token test for memorization.",
        "row-independence": "Row independence test, using a gradient boosted tree and logistic regression.",
    }

    parser.add_argument("csv", type=str, help="A csv file.")
    parser.add_argument(
        "task",
        type=str,
        choices=list(tasks.keys()),
        help="The task that should be performed with the csv file.\n  - "
        + "\n  - ".join([k + ": " + v for k, v in tasks.items()]),
        metavar="task",
    )

    # test parameters
    parser.add_argument("--header-length", type=int, default=500)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-prefix-rows", type=int, default=None)
    parser.add_argument("--num-prefix-features", type=int, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument(
        "--few-shot",
        "--names-list",
        nargs="*",
        default=[
            "csv/IRIS.csv",
            "csv/adult.csv",
            "csv/titanic-train.csv",
            "csv/uci-wine.csv",
            "csv/california-housing.csv",
        ],
    )
    # the --cond parameter takes a list of strings
    parser.add_argument("--cond", nargs="*", default=[])

    # openai api args
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--engine", type=str, default=None)

    # LLM
    parser.add_argument("--temperature", type=float, default=0.0)

    # misc args
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Where to save the results (a filename).",
    )
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--pp", default=False, action="store_true"
    )  # print prompts & responses
    parser.add_argument(
        "--pr", default=False, action="store_true"
    )  # print responses only

    args = parser.parse_args()

    # debug mode
    if args.debug:
        args.pp = True
        args.pr = True

    # setup. this takes care of the lower-level args
    experiment_utils.setup(args)

    # debug mode: print the first query
    if args.debug:
        experiment_utils.print_next_prompt = True

    # load the default config file
    with open("system-prompts.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # args.csv can be a single csv file or a directory
    csv_files = [args.csv]
    if os.path.isdir(args.csv):
        csv_files = [
            os.path.join(args.csv, file)
            for file in os.listdir(args.csv)
            if file.endswith(".csv")
        ]

    # --few-shot can be an integer or a list of csv files
    few_shot = args.few_shot
    if few_shot is not None:
        if len(few_shot) == 1:
            try:  # attempt to convert to int. if it fails, assume it is a csv file
                few_shot = int(few_shot[0])
            except ValueError:
                pass

    # run the specified task for all the csv files
    for csv_file in csv_files:
        print(
            experiment_utils.bcolors.BOLD
            + f"File: "
            + experiment_utils.bcolors.ENDC
            + f"{csv_file}"
        )
        # make sure that the current csv file is not contained in the few-shot list
        csv_file_few_shot = copy.deepcopy(few_shot)
        if isinstance(few_shot, list):
            csv_file_few_shot = [
                file
                for file in few_shot
                if not utils.get_dataset_name(csv_file) in file
            ]
            if len(csv_file_few_shot) < len(few_shot):
                middle = len(csv_file_few_shot) // 2
                csv_file_few_shot = (
                    csv_file_few_shot[:middle]
                    + ["csv/openml-diabetes.csv"]
                    + csv_file_few_shot[middle:]
                )
                print(
                    "INFO: The csv file was contained in the few-shot list, replaced it with openml-diabetes.csv"
                )

        if args.task == "feature-names":
            if args.num_prefix_features is None:
                args.num_prefix_features = len(utils.get_feature_names(csv_file)) // 3
            test_functions.feature_names_test(
                csv_file,
                config["feature-names"],
                num_prefix_features=args.num_prefix_features,
                few_shot_csv_files=csv_file_few_shot,
            )
        elif args.task == "sample":
            test_functions.sample(
                csv_file,
                config["sample"],
                num_queries=args.num_queries,
                few_shot=csv_file_few_shot,
                cond_feature_names=args.cond,
                out_file=args.out,
            )
        elif args.task == "completion":
            test_functions.conditional_completion_test(
                csv_file,
                config["sample"],
                feature_name=args.target,
                num_queries=args.num_queries,
                few_shot=csv_file_few_shot,
                out_file=args.out,
            )
        elif args.task == "predict":
            # set the callback
            # experiment_utils.on_response_callback_fn = analysis.csv_match_callback(
            #    csv_file
            # )
            test_functions.predict(
                csv_file,
                config["predict"],
                args.target,
                num_queries=args.num_queries,
                # TODO few_shot=csv_file_few_shot,
            )
        elif args.task == "row-independence":
            row_independence_test(csv_file)
        elif args.task == "format":
            test_functions.csv_format_test(
                csv_file, system_prompt=config["generic-csv-format"]
            )
        elif args.task == "header":
            test_functions.header_test(
                csv_file, config["header-test"], few_shot_csv_files=csv_file_few_shot
            )
        elif args.task == "row-completion":
            if args.num_prefix_rows is None:
                args.num_prefix_rows = 15
            if args.few_shot is None or isinstance(args.few_shot, list):
                args.few_shot = 7
            test_functions.row_completion_test(
                csv_file,
                config["row-completion"],
                num_queries=args.num_queries,
                num_prefix_rows=args.num_prefix_rows,
                few_shot=args.few_shot,
                out_file=args.out,
            )
        elif args.task == "first-token":
            if args.num_prefix_rows is None:
                args.num_prefix_rows = 15
            if args.few_shot is None or isinstance(args.few_shot, list):
                args.few_shot = 7
            # we should use row-completion with max-tokens?
            test_functions.first_token_test(
                csv_file,
                config["row-completion"],
                num_queries=args.num_queries,
                num_prefix_rows=args.num_prefix_rows,
                few_shot=args.few_shot,
                save_path="results/",
            )
        elif args.task == "ordered-completion":
            test_functions.ordered_completion(
                csv_file,
                config["feature-completion"],
                feature_name=args.target,
                num_queries=args.num_queries,
                few_shot=csv_file_few_shot,
                out_file=args.out,
            )
        elif args.task == "feature-completion":
            test_functions.feature_completion_test(
                csv_file,
                config["feature-completion"],
                feature_name=args.target,
                num_queries=args.num_queries,
                out_file=args.out,
            )

    exit(0)
