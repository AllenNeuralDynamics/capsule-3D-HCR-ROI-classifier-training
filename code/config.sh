#!/usr/bin/env bash

if [ -z "${1}" ]; then
  alpha=0.5
else
  alpha="${1}"
fi

if [ -z "${2}" ]; then
  l1_ratio=0.5
else
  l1_ratio="${2}"
fi

if [ -z "${3}" ]; then
  s3_bucket="s3://research-machine/mlflow"
else
  s3_bucket="${3}"
fi

if [ -z "${4}" ]; then
  input_data="../data/diabetes_raw_data.csv"
else
  input_data="${4}"
fi