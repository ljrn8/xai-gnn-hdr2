"""Top level script for running mupltiple explainer configurations on all target datasets and available models"""

from run_explainers import main as expl_main
from evaluate_explainers import main as eval_main
from argparse import ArgumentParser
from loguru import logger
import os
from pathlib import Path


def run_explanation_pipeline(
    args,
    explainer_configurations={
        "learning_rate": [0.01, 0.05, 0.1],
        "hidden_size": [64, 64, 32],
        "epochs": [300, 200, 100],
    },
):
    for i in range(len(explainer_configurations.values()[0])):
        for ds_name in os.listdir(args["root"]):
            logger.info(
                f"\n\n === BEGAN Running Explainer Configurations for dataset: {ds_name} === \n"
            )
            expl_main({""})


def main(args):
    if args.eval_instead:
        ...
    else:
        run_explanation_pipeline(args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-r", "--root", required=True, help="root directory of multiple dataset folders"
    )
    parser.add_argument(
        "-e", "--eval-instead", action="store_true", help="evaluate explainers instead"
    )
    main(parser.parse_args())
