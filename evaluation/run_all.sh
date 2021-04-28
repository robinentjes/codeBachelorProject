for dataset in rw men simlex wordsim
do
  for size in 128 256 512
  do
    for model in nll ste
    do
      echo "Dataset: $dataset, Size: $size, Model: $model"
      python evaluation.py $dataset $size $model
    done
  done
done