#
# All sorts of functions used in the experiments.
#

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from datetime import datetime


import copy
import pickle
import openai
import os
import glob

import promptutil
import send_model_messages


####################################################################################
# Tabular chat completion fit-predict function
####################################################################################


def tabular_chat_fit_predict_fn_factory(feature_names, target_name, messages, **kwargs):
    # create a safe copy of the initial messages
    messages = copy.deepcopy(messages)

    # the fit_predict function that we return to the user
    def fit_predict(X_train, y_train, testpoint):
        prediction = send_tabular_chat_completion(
            X_train,
            y_train,
            testpoint,
            feature_names,
            target_name,
            copy.deepcopy(messages),
            **kwargs,
        )
        return 0  # TODO: return the prediction but in guarnateed format?

    return fit_predict


####################################################################################
# Send chat completion messages for tabular data
####################################################################################


def send_tabular_chat_completion(
    X_train,
    y_train,
    testpoint,
    feature_names,
    target_name,
    messages,
    optional_features=None,
    logfile=None,
    temperature=0.0,
    max_tokens=100,
):
    for idx in range(X_train.shape[0]):
        messages.append(
            {
                "role": "user",
                "content": promptutil.format_data_point(
                    X_train[idx, :],
                    feature_names,
                    optional_features=optional_features,
                    add_if_then=len(feature_names) != 0,
                ),
            }
        )
        # add target name if it is provided as input
        if len(target_name) > 0:
            messages.append(
                {"role": "assistant", "content": f"{target_name} = {y_train[idx]}"}
            )
        else:
            messages.append({"role": "assistant", "content": f"{y_train[idx]}"})
    messages.append(
        {
            "role": "user",
            "content": promptutil.format_data_point(
                testpoint,
                feature_names,
                optional_features=optional_features,
                add_if_then=len(feature_names) != 0,
            ),
        }
    )

    response = send_chat_completion(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        logfile=logfile,
    )

    try:  # does not work in batch mode
        return response["choices"][0]["message"]["content"]
    except:
        return 0


####################################################################################
# Send chat completion messages with retrying and logging.
#
# The logging works as follows. Either we specify a logfile, then we log to that file.
# Or we specify a global task, then we log to taskname/taskname-{idx}.pkl where idx is the number of the
# query under that task. The task will typically be the name of the dataset
# and idx the index of the testpoint. If we specify neither, we log to a timestamp.
#
####################################################################################
current_logging_task = None
current_logging_folder = None
logging_task_index = 0


def set_logging_task(task):
    """Set the global task for logging."""
    global current_logging_task
    global current_logging_folder
    global logging_task_index

    current_logging_task = task
    current_logging_folder = f"chatlogs/{current_logging_task}"
    logging_task_index = 0

    # check if the folder exists, if not create it
    if not os.path.exists(current_logging_folder):
        os.makedirs(current_logging_folder)


def read_chatlog(taskname, root="chatlogs", min_files=-1):
    """A chaglog is a sequnces of files 'taskname-{idx}.pkl' that contain message-response pairs"""
    messages = []
    responses = []

    task_dir = os.path.join(root, taskname)
    # list all the files in task_dir
    task_files = os.listdir(task_dir)
    for idx in range(10000):
        # fname = f"{taskname}-{idx}.pkl"
        fsuffix = f"-{idx}.pkl"
        fname = [f for f in task_files if f.endswith(fsuffix)]
        if len(fname) > 1:
            print(f"Warning: found more than one file with suffix {fsuffix}")
        if len(fname) > 0:
            fname = fname[0]
            try:
                with open(os.path.join(task_dir, fname), "rb") as f:
                    message, response = pickle.load(f)
                    messages.append(message)
                    responses.append(response)
            except:
                print(f"Failed to read {fname}")
                messages.append(None)
                responses.append(None)
        elif (
            len(messages) > min_files
        ):  # if we already have enough results, then this is the end
            break
        else:
            print(f"File {fname} not found.")
            messages.append(None)
            responses.append(None)
    return messages, responses


@retry(
    retry=retry_if_not_exception_type(openai.error.InvalidRequestError),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(10),
)
def send_chat_completion(messages, temperature, max_tokens, logfile=None):
    global current_logging_task
    global logging_task_index

    """Send chat completion with retrying and logging."""
    response = send_model_messages.send_chat_completion(
        messages, temperature, max_tokens
    )
    # logging
    if logfile is None:
        if current_logging_task is not None:
            logfile = f"{current_logging_task}/{current_logging_task}-{logging_task_index}.pkl"
            logging_task_index += 1
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f")
            logfile = timestamp + ""
    with open(f"chatlogs/{logfile}", "wb+") as f:
        pickle.dump((messages, response), f)
    # return full response
    return response
