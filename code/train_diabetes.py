#
# train_diabetes.py
#
#   MLflow model using ElasticNet (sklearn) and Plots ElasticNet Descent Paths
#
#   Uses the sklearn Diabetes dataset to predict diabetes progression using ElasticNet
#       The predicted "progression" column is a quantitative measure of disease progression one year after baseline
#       http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
#   Combines the above with the Lasso Coordinate Descent Path Plot
#       http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html
#       Original author: Alexandre Gramfort <alexandre.gramfort@inria.fr>; License: BSD 3 clause
#
#  Usage:
#    python train_diabetes.py 0.01 0.01
#    python train_diabetes.py 0.01 0.75
#    python train_diabetes.py 0.01 1.0
#

import argparse
import os
import warnings
import sys

import pandas as pd
from pathlib import Path
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
import shlex
import shutil
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets
from subprocess import check_call, CalledProcessError
import mlflow 
import logging

# Import mlflow
import mlflow
import mlflow.sklearn


logging.basicConfig(format='%(asctime)s - %(name)s - {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Evaluate metrics
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def main(argv=None): 
    parser = argparse.ArgumentParser(description="Generate model of prediction of wine preferences based on physiochemical properties")
    parser.add_argument("--s3_bucket", help="AWS s3 location for artifacts", default="s3://research-machine/mlflow")
    parser.add_argument("--experiment_name", help="Name for mlflow experiment", default="diabetes_experiment")
    parser.add_argument("--input_data", help="Input data to train model.", default="../data/diabetes_raw_data.csv")
    parser.add_argument("--alpha", help="alpha parameter for ElasticNet model", type=float, default=0.5)
    parser.add_argument("--l1_ratio", help="l1_ratio parameter for ElasticNet model", type=float, default=0.5)
    if argv:
        args = parser.parse_args(argv)
    else: 
        parser.print_usage()
        return 0
    os.system(f"mkdir /tmp/mlflow")
    os.system(f"aws s3 sync {args.s3_bucket}/experiments /tmp/mlflow/db --quiet")
    mlflow.set_tracking_uri("/tmp/mlflow/db")

    #create experiment if it doesn't already exist. 
    existing_exp = mlflow.get_experiment_by_name(args.experiment_name)
    if not mlflow.get_experiment_by_name(args.experiment_name):
        mlflow.create_experiment(args.experiment_name, args.s3_bucket)
    mlflow.set_experiment(args.experiment_name)
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    data = pd.read_csv(args.input_data)
    
    y = np.array(data['progression'])
    X = np.array(data.drop('progression', axis=1))

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "progression" which is a quantitative measure of disease progression one year after baseline
    train_x = train.drop(["progression"], axis=1)
    test_x = test.drop(["progression"], axis=1)
    train_y = train[["progression"]]
    test_y = test[["progression"]]
    
    #alpha = 0.05
    #l1_ratio = 0.01

    # Run ElasticNet
    lr = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # Print out ElasticNet model metrics
    logger.info("Elasticnet model (alpha=%f, l1_ratio=%f):" % (args.alpha, args.l1_ratio))
    logger.info("  RMSE: %s" % rmse)
    logger.info("  MAE: %s" % mae)
    logger.info("  R2: %s" % r2)

    # Log mlflow attributes for mlflow UI
    mlflow.log_param("alpha", args.alpha)
    mlflow.log_param("l1_ratio", args.l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(lr, "model")

    # Compute paths
    eps = 5e-3  # the smaller it is the longer is the path

    logger.info("Computing regularization path using the elastic net.")
    alphas_enet, coefs_enet, _ = enet_path(X, y, eps=eps, l1_ratio=args.l1_ratio)

    # Display results
    fig = plt.figure(1)
    ax = plt.gca()

    colors = cycle(["b", "r", "g", "c", "k"])
    neg_log_alphas_enet = -np.log10(alphas_enet)
    for coef_e, c in zip(coefs_enet, colors):
        l2 = plt.plot(neg_log_alphas_enet, coef_e, linestyle="--", c=c)

    plt.xlabel("-Log(alpha)")
    plt.ylabel("coefficients")
    title = "ElasticNet Path by alpha for l1_ratio = " + str(args.l1_ratio)
    plt.title(title)
    plt.axis("tight")

    # Save figures
    fig.savefig("../results/ElasticNet-paths.png")

    # Close plot
    plt.close(fig)

    # Log artifacts (output files)
    mlflow.log_artifact("../results/ElasticNet-paths.png")
    os.system(f"aws s3 sync /tmp/mlflow/db {args.s3_bucket}/experiments --quiet")
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))