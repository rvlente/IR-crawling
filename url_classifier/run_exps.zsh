#!/urs/bin/env zsh

emulate -LR zsh
set -e

n_estimators=(500)
top_ks=(500)
ngram_sizes=(1 2 3 4 5)


path_to_current_dir=$(dirname $(realpath $0))
script_path=${path_to_current_dir}/url_classifier.py
data_path=${path_to_current_dir}/../cache/url_data2.csv

export MLFLOW_EXPERIMENT_NAME="script_exps"

for n_estimator in ${n_estimators[@]}; do
    for top_k in ${top_ks[@]}; do
        for ngram_size in ${ngram_sizes[@]}; do
            echo "n_estimator: $n_estimator, top_k: $top_k, ngram_size: $ngram_size"
            python3 $script_path \
                --n-estimators $n_estimator \
                --top-k-ngrams $top_k \
                --ngram-size $ngram_size \
                --datafile $data_path
        done
    done
done