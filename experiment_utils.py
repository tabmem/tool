#
# All sorts of functions used in the experiments.
#

from datetime import datetime

import os
import pickle
import time

import openai_api

import tiktoken


####################################################################################
# Setup function (logging, prompt printing, openai api, etc.)
####################################################################################


# global config variables
print_prompts = False
print_responses = False
print_next_prompt = False

llm_temperature = None
llm_max_tokens = 500
sleep_in_between_llm_queries = 0.0

# callback function that is called every time a chat response is received
# used to print analysis in the chat directly after a response comes in
#
# called as on_response_callback_fn(messages, response)
#
# returns a stirng that is printed in the chat directly after the response (given that we have response printing enabled)
#
on_response_callback_fn = None


def setup(args):
    global print_prompts
    global llm_temperature
    global sleep_in_between_llm_queries
    global print_responses

    # print prompts
    if args.pp:
        print_prompts = True
        print_responses = True

    # print respnses
    if args.pr:
        print_responses = True

    llm_temperature = args.temperature
    sleep_in_between_llm_queries = args.sleep

    # openai api setup
    openai_api.openai_api_setup(model=args.model, engine=args.engine)


####################################################################################
# Send chat completion messages with retrying and logging.
#
# The logging works as follows. Either we specify a logfile, then we log to that file.
# Or we specify a global task, then we log to taskname-{idx} where idx is the number of the
# query under that task. The task will typically be the name of the dataset
# and idx the index of the testpoint.
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
        #fname = f"{taskname}-{idx}.pkl"
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


def send_chat_completion(messages, max_tokens=None, logfile=None):
    """Send chat completion with retrying and logging."""
    global current_logging_task
    global logging_task_index
    global print_prompts, print_responses, print_next_prompt
    global llm_temperature, llm_max_tokens
    global on_response_callback_fn

    if max_tokens is None:
        max_tokens = llm_max_tokens

    response = openai_api.send_chat_completion(messages, llm_temperature, max_tokens)
    if sleep_in_between_llm_queries > 0.0:
        time.sleep(sleep_in_between_llm_queries)
    # logging
    if logfile is None:
        if current_logging_task is not None:
            logfile = f"{current_logging_task}/{current_logging_task}-{logging_task_index}.pkl"
            logging_task_index += 1
    if logfile is not None:
        with open(f"chatlogs/{logfile}", "wb+") as f:
            pickle.dump((messages, response), f)
    # printing
    if print_prompts or print_next_prompt:
        pretty_print_messages(messages)
    if print_responses or print_next_prompt:
        pretty_print_response(response["choices"][0]["message"]["content"])
    # callback with optional printing
    if on_response_callback_fn is not None:
        s = on_response_callback_fn(
            messages, response["choices"][0]["message"]["content"]
        )
        if s is not None and len(s) > 0:
            print(bcolors.Blue + s + bcolors.ENDC)
    # reset print_next_prompt
    print_next_prompt = False
    # return openai response object
    return response


#################################################################################################
# misc
#################################################################################################

import json


def pretty_print_messages(messages):
    """Prints openai chat messages in a nice format"""
    for message in messages:
        print(
            bcolors.BOLD
            + message["role"]
            + ": "
            + bcolors.ENDC
            + bcolors.Green
            + message["content"].strip()
            + bcolors.ENDC
        )


def pretty_print_response(response):
    """Prints openai chat response in a nice format"""
    print(
        bcolors.BOLD
        + "Response: "
        + bcolors.ENDC
        + bcolors.Purple
        + response
        + bcolors.ENDC,
    )


def num_tokens_from_string(string: str, model_name: str = None) -> int:
    """Returns the number of tokens in a text string."""
    # if the user did not specify the encoding, take the maximum over gpt 3.5 and gpt 4
    # TODO in the future do this for the specified llm, once we have it in the codebase
    if model_name is None:
        encoding = tiktoken.encoding_for_model("gpt-4")
        num_tokens_gpt_4 = len(encoding.encode(string))
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        num_tokens_gpt_3_5 = len(encoding.encode(string))
        return max(num_tokens_gpt_4, num_tokens_gpt_3_5)
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


#################################################################################################
# color codes to print with color in the console (from https://gist.github.com/vratiu/9780109)
#################################################################################################


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    # Regular Colors
    Black = "\033[0;30m"  # Black
    Red = "\033[0;31m"  # Red
    Green = "\033[0;32m"  # Green
    Yellow = "\033[0;33m"  # Yellow
    Blue = "\033[0;34m"  # Blue
    Purple = "\033[0;35m"  # Purple
    Cyan = "\033[0;36m"  # Cyan
    White = "\033[0;37m"  # White

    # Background
    On_Black = "\033[40m"  # Black
    On_Red = "\033[41m"  # Red
    On_Green = "\033[42m"  # Green
    On_Yellow = "\033[43m"  # Yellow
    On_Blue = "\033[44m"  # Blue
    On_Purple = "\033[45m"  # Purple
    On_Cyan = "\033[46m"  # Cyan
    On_White = "\033[47m"  # White
