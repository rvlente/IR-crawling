#!/urs/bin/env zsh

emulate -LR zsh
set -e

n_estimators=(100 200 300 400 500)
top_ks=(100 200 300 400 500)
ngram_sizes=(1 2 3 4 5)

exp_name="$(date -Iseconds)"
path_to_current_dir=$(dirname $(realpath $0))
script_path=${path_to_current_dir}/url_classifier.py
data_path=${path_to_current_dir}/../data/url_data_full.csv

mlflow experiments create -n $exp_name

args_file=$(mktemp /tmp/args.XXXXXX)

for n_estimator in ${n_estimators[@]}; do
    for top_k in ${top_ks[@]}; do
        for ngram_size in ${ngram_sizes[@]}; do
            # echo "n_estimator: $n_estimator, top_k: $top_k, ngram_size: $ngram_size"
            # args+=(--n_estimator $n_estimator --top_k $top_k --ngram_size $ngram_size --datafile $data_path --exp-name $exp_name;)
            echo "--n-estimators $n_estimator -k $top_k --ngram-size $ngram_size --datafile $data_path --exp-name $exp_name" >> $args_file
            # python3 $script_path \
            #     --n-estimators $1 \
            #     --top-k-ngrams $top_k \
            #     --ngram-size $ngram_size \
            #     --datafile $data_path \
            #     --exp-name $exp_name 
        done
    done
done

# Run script in parallel
xargs -a $args_file -n 10 -P ${1:-2} python3 $script_path

rm $args_file
