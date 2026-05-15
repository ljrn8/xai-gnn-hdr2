. env/Scripts/activate 
set -e

echo " * Explaining node-level models \n"
for f in output/node_level/*; do
    echo "\n * Training dataset: $f \n"
    python -m explainability.run_explainers --ds-root $f --node-level
    echo "\n * Evaluating dataset: $f \n"
    python -m explainability.evaluate --ds-root $f --node-level
done

echo "\n * Explaining graph-level models \n"
for f in output/graph_level/*; do
    echo "\n * Training dataset: $f \n"
    python -m explainability.run_explainers --ds-root $f --graph-level
    echo "\n * Evaluating dataset: $f \n"
    python -m explainability.evaluate --ds-root $f --graph-level
done