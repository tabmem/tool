# "Elephants Never Forget: Testing Language Models for Memorization of Tabular Data" (Anonymous Code Repository)

This anonymous code repository contains a command line tool to conduct the different tests that are described in the paper. It also contains the Jupyter notebooks that generate the figures and tables in the paper.

The command line tool can be called as <pre>python csv-llm-memtool.py</pre> The usage of the tool is as follows

<pre>
A python tool for csv files and large language models. Provides different memorization tests. Use --debug to see the prompts.

positional arguments:
  csv                   A csv file.
  task                  The task that should be performed with the csv file.
                          - predict: Few-shot prediction of a feature in the csv file. Use --target to specify the target feature.
                          - sample: Ask the LLM to provide random samples from the csv file. For conditional sampling, use the parameter --cond.
                          - feature-names: Does the LLM to complete the feature names from the top row of the csv file?
                          - ordered-completion: Feature completion, respecting the order of the features in the csv file.
                          - header: Header test for memorization.
                          - row-completion: Row completion test for memorization.
                          - feature-completion: Feature completion test for memorization.
                          - first-token: First token test for memorization.
                          - row-independence: Row independence test, using a gradient boosted tree and logistic regression.

optional arguments:
  -h, --help            show this help message and exit
  --header-length HEADER_LENGTH
  --num-queries NUM_QUERIES
  --num-prefix-rows NUM_PREFIX_ROWS
  --num-prefix-features NUM_PREFIX_FEATURES
  --target TARGET
  --few-shot [FEW_SHOT ...], --names-list [FEW_SHOT ...]
  --cond [COND ...]
  --model MODEL
  --engine ENGINE
  --temperature TEMPERATURE
  --out OUT             Where to save the results (a filename).
  --sleep SLEEP         Sleep SLEEP seconds in between queries.
  --debug               Print all the prompts, including model responses.
  --pr                  Print only the model responses.

</pre>

All the results in the paper were generated using this tool. By default, the tool uses gpt-3.5-turbo-16k-0613. To use a different LLM, use the --model parameter. We used the following commands for the paper.

<pre>
# conditional completion on Adult Income
features=(
    Education
    EducationNum
    Occupation
    Gender
    CapitalLoss
    HoursPerWeek
    NativeCountry
    Income
)
for i in "${features[@]}"; do
    python csv-llm-memtool.py csv/adult.csv ordered-completion --target $i --num-queries 250 --temperature 0.0 --out results/gpt-3.5-turbo/ordered-completion/adult-$i.csv
done
  
# conditional completion on FICO
features=(
    AverageMInFile
    NumTrades90Ever2DerogPubRec
    MaxDelq2PublicRecLast12M
    NumTradesOpeninLast12M
    NumInqLast6M
    NetFractionInstallBurden
    NumBank2NatlTradesWHighUtilization
    PercentTradesWBalance
)
for i in "${features[@]}"; do
    python csv-llm-memtool.py ../private-do-not-distribute/fico.csv ordered-completion --target $i --num-queries 250 --temperature 0.0 --out ../private-do-not-distribute/results/fico-ordered-completion-$i.csv
done

# Figure 2 in the main paper
python csv-llm-memtool.py csv/california-housing.csv sample --num-queries 1000 --out results/housing-samples-0.2.csv --temperature 0.2
python csv-llm-memtool.py csv/california-housing.csv sample --num-queries 1000 --out results/housing-samples-0.6.csv --temperature 0.6
python csv-llm-memtool.py csv/california-housing.csv sample --num-queries 1000 --out results/housing-samples-1.2.csv --temperature 1.2

# zero-knowledge samples with temperature 0.7
datasets=(
    IRIS
    uci-wine
    sklearn-diabetes
    titanic-train
    openml-diabetes
    adult
    california-housing
)
for i in "${datasets[@]}"; do
    python csv-llm-memtool.py csv/$i.csv sample --num-queries 1000 --temperature 0.7 --out results/gpt-3.5-turbo/samples/$i-temperature-0.7.csv
done
  
# Header Test
python csv-llm-memtool.py csv/IRIS.csv header
python csv-llm-memtool.py csv/uci-wine.csv header
python csv-llm-memtool.py csv/heart.csv header
python csv-llm-memtool.py csv/titanic-train.csv header
python csv-llm-memtool.py csv/openml-diabetes.csv header
python csv-llm-memtool.py csv/sklearn-diabetes.csv header
python csv-llm-memtool.py csv/adult.csv header
python csv-llm-memtool.py csv/california-housing.csv header
python csv-llm-memtool.py csv/spaceship-titanic-train.csv header
python csv-llm-memtool.py ../private-do-not-distribute/FICO.csv header
#python csv-llm-memtool.py ../private-do-not-distribute/pneumonia.csv header

# row completion
datasets=(
    IRIS
    uci-wine
    sklearn-diabetes
    titanic-train
    openml-diabetes
    adult
    california-housing
    spaceship-titanic-train
)
for i in "${datasets[@]}"; do
    python csv-llm-memtool.py csv/$i.csv row-completion --num-queries 250 --temperature 0.0 --out results/gpt-3.5-turbo/row-completion/$i.csv
done

python csv-llm-memtool.py ../private-do-not-distribute/fico.csv row-completion --num-queries 250 --temperature 0.0 --out ../private-do-not-distribute/results/fico-row-completion.csv
python csv-llm-memtool.py ../private-do-not-distribute/pneumonia.csv row-completion --num-queries 50 --temperature 0.0 --out ../private-do-not-distribute/results/pneumonia-row-completion-gpt4.csv --model=gpt-4 --num-prefix-rows  7  --few-shot 5         

# feature completion
python csv-llm-memtool.py csv/uci-wine.csv feature-completion --num-queries 250 --temperature 0.0 --out results/gpt-3.5-turbo/feature-completion/uci-wine.csv
python csv-llm-memtool.py csv/sklearn-diabetes.csv feature-completion --num-queries 250 --temperature 0.0 --out results/gpt-3.5-turbo/feature-completion/sklearn-diabetes.csv
python csv-llm-memtool.py csv/openml-diabetes.csv feature-completion --num-queries 250 --temperature 0.0 --out results/gpt-3.5-turbo/feature-completion/openml-diabetes.csv
python csv-llm-memtool.py csv/titanic-train.csv feature-completion --target name --num-queries 250 --temperature 0.0 --out results/gpt-3.5-turbo/feature-completion/titanic-train.csv
python csv-llm-memtool.py csv/adult.csv feature-completion --target fnlwgt --num-queries 250 --temperature 0.0 --out results/gpt-3.5-turbo/feature-completion/adult.csv
python csv-llm-memtool.py csv/california-housing.csv feature-completion --num-queries 250 --temperature 0.0 --out results/gpt-3.5-turbo/feature-completion/california-housing.csv
python csv-llm-memtool.py ../private-do-not-distribute/fico.csv feature-completion --num-queries 250 --temperature 0.0 --out ../private-do-not-distribute/results/fico-feature-completion.csv
python csv-llm-memtool.py ../private-do-not-distribute/pneumonia.csv feature-completion  --num-queries 250 --temperature 0.0 --out ../private-do-not-distribute/results/pneumonia-feature-completion.csv
python csv-llm-memtool.py csv/spaceship-titanic-train.csv feature-completion --target Name --num-queries 250 --temperature 0.0 --out results/gpt-3.5-turbo/feature-completion/spaceship-titanic-train.csv
</pre>


