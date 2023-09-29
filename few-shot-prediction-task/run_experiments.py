#
# Run the experiments!
#
# Call as 'python run_experiments.py'
#

import pandas as pd
import yaml

from sklearn.model_selection import train_test_split

import promptutil
import experiment_utils


# main entry point
if __name__ == "__main__":
    print("Starting the experiments.")

    ####################################################################################################
    # chat model leave-one-out evaluation with tabular data
    ####################################################################################################

    experiments = {
        "adult": {
            "csv-file": "datasets/csv/tabular/adult.csv",
            "config": "config/tabular/adult.yaml",
         },
         "openml-diabetes": {
            "csv-file": "datasets/csv/tabular/openml-diabetes.csv",
            "config": "config/tabular/openml-diabetes.yaml",
         },
        "spaceship-titanic": {
            "csv-file": "datasets/csv/tabular/spaceship-titanic-train.csv",
            "config": "config/tabular/spaceship-titanic.yaml",
        },
    }

    for exp_name, experiment in experiments.items():
        # read parameters from experiment
        print("Running experiment: ", exp_name)
        print("Experiment parameters: ", experiment)

        # load the csv file
        df = pd.read_csv(experiment["csv-file"])

        # load the config file
        with open(experiment["config"], "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # the names of the features and the target
        feature_names, target_name = df.columns.tolist()[:-1], df.columns.tolist()[-1]
        if "features" in config:
            feature_names = config["features"]
        if "target" in config:
            target_name = config["target"]

        # features that are optional in the prompt. These will only be included if they are not zero. (used to handle rare diseases in the pneumonia dataset)
        optional_features = []
        if "optional_features" in config:
            optional_features = config["optional_features"]

        # recode the values of features in the data (usually to replace them with strings that the LLM can understand)
        if "recode_features" in config:
            for feature_name, recode_dict in config["recode_features"].items():
                df[feature_name] = df[feature_name].replace(recode_dict)

        # the data
        X_data, y_data = df[feature_names].values, df[target_name].values

        # for big datasets, perform a train-test split. this also shuffles the data which is important
        if X_data.shape[0] > 2500:
            X_train, _, y_train, _ = train_test_split(
                X_data, y_data, test_size=0.2, random_state=42
            )
        else:
            X_train, y_train = X_data, y_data

        # the system prompt
        system_prompt = config["system_prompt"]

        # remove leading and trailing whitespaces from feature and target names
        feature_names = [x.strip() for x in feature_names]
        target_name = target_name.strip()

        # detect if the name of the target in the csv is missing. If yes, set to empty string (this will remove the target name from the chat response)
        if target_name[:8] == "Unnamed:" or (
            "use_target_name" in config and config["use_target_name"] == False
        ):
            target_name = ""

        # option not to use the feature names, used for customt prompt input
        if "use_feature_names" in config and config["use_feature_names"] == False:
            feature_names = []

        # setup for running the experiment
        experiment_utils.set_logging_task(exp_name)

        # run the experiment
        messages = [{"role": "system", "content": system_prompt}]
        fit_predict_fn = experiment_utils.tabular_chat_fit_predict_fn_factory(
            feature_names, target_name, messages, optional_features=optional_features
        )

        promptutil.loo_eval(
            X_train, y_train, fit_predict_fn, few_shot=20, max_points=1000
        )

    print("Experiments are done, this has been fun, is this is a pun?")
    print("All has now ended, yes pun intended.")
