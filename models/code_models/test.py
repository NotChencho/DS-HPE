import torch
import sys
from pathlib import Path


RootPath = Path(__file__).resolve().parents[2]

DataPath = RootPath / "data" / "interm"

def main():

    train_ds = torch.load(DataPath / "train_dataset.pt")
    print(train_ds)


if __name__ == "__main__":

    main()