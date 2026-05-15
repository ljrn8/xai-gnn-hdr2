. env/Scripts/activate

echo "\n * Running node-level models \n"
for f in output/node_level/*; do
    echo "\n * Training dataset: $f \n"
    python -m training.train_models --ds-root $f --node-level --model-configurations "transductive citation datasets" -i 10
done

echo "\n * Running graph-level models \n"
for f in output/graph_level/*; do
    echo "\n * Training dataset: $f \n"
    python -m training.train_models --ds-root $f --graph-level --model-configurations "inductive small graphs" -i 4
done 

echo "\n * Eval for node-level models \n"
for f in output/node_level/*; do
    echo "\n * Evaluating dataset: $f \n"
    python -m training.evaluate --ds-root $f --node-level --model-configurations "transductive citation datasets" -i 10
done

echo "\n * Eval for graph-level models \n"
for f in output/graph_level/*; do
    echo "\n * Evaluating dataset: $f \n"
    python -m training.evaluate --ds-root $f --graph-level --model-configurations "inductive small graphs" -i 4
done