import os

folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "models",
    "dashboard",
    "reports/charts",
    "reports/insights",
    "tests"
]

files = [
    "notebooks/01_eda.ipynb",
    "notebooks/02_feature_engineering.ipynb",
    "src/data_cleaning.py",
    "src/feature_engineering.py",
    "src/train.py",
    "src/evaluate.py",
    "src/forecast.py",
    "dashboard/app.py",
    "models/random_forest.pkl",
    "requirements.txt",
    "README.md",
    ".gitignore"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for file in files:
    with open(file, "w") as f:
        if file.endswith(".ipynb"):
            f.write('{\n "cells": [],\n "metadata": {},\n "nbformat": 4,\n "nbformat_minor": 5\n}')
        else:
            f.write("")

print("Project structure created successfully!")