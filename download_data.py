
import os, sys, pathlib, textwrap

DATA_PATH = pathlib.Path('data/creditcard.csv')

def main():
    if DATA_PATH.exists():
        print(f'[ok] Found dataset at {DATA_PATH}')
        return
    print(textwrap.dedent(f"""
    The dataset 'creditcard.csv' is not present.
    1) Download the 'Credit Card Fraud Detection' CSV (often mirrored as ULB/Kaggle dataset).
    2) Place it at: {DATA_PATH.resolve()}
    3) Re-run your training command.
    """))

if __name__ == '__main__':
    main()
