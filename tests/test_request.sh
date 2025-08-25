#!/bin/bash

curl -s -X POST http://localhost:8000/predict_path \
  -H 'Content-Type: application/json' \
  -d '{"csv_path": "datasets/heart_test.csv"}'
