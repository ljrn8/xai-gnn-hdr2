. env/Scripts/activate

# echo "\n * Running node-level models \n"
# for f in output/node_level/*; do
#     echo "\n * Training dataset: $f \n"
#     python -m training.train_models --ds-root $f --node-level --model-configurations "transductive citation datasets" -i 10
# done

# echo "\n * Running graph-level models \n"
# for f in output/graph_level/*; do
#     echo "\n * Training dataset: $f \n"
#     python -m training.train_models --ds-root $f --graph-level --model-configurations "inductive small graphs" -i 4
# done 

echo " * Eval for node-level models "
for f in output/node_level/*; do
    echo " * Evaluating dataset: $f "
    python -m training.evaluate --ds-root $f --node-level 
done

echo " * Eval for graph-level models "
for f in output/graph_level/*; do
    echo " * Evaluating dataset: $f "
    python -m training.evaluate --ds-root $f --graph-level 
done