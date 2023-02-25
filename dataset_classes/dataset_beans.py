import argparse
import csv
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import load_dataset
from pathlib import Path

import torchvision.transforms as tv

# ----- core transforms ----------

_imgnet_normalize = tv.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transforms = tv.Compose(
    [
        tv.Resize((256, 256)),
        tv.RandomResizedCrop(224),
        tv.RandomHorizontalFlip(),
        tv.ToTensor(),
        _imgnet_normalize,
    ]
)

validation_transforms = tv.Compose(
    [
        tv.Resize((256, 256)),
        tv.CenterCrop(224),
        tv.ToTensor(),
        _imgnet_normalize,
    ]
)

# ---- functions ---------


def train_batch_transforms(samples):
    samples["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in samples["image"]
    ]
    del samples["image"]
    return samples


def validation_batch_transforms(samples):
    samples["pixel_values"] = [
        validation_transforms(image.convert("RGB")) for image in samples["image"]
    ]
    del samples["image"]
    return samples


def get_datasets():
    """cover function for handling loading the training and validation dataset"""
    """dataset loading"""
    from datasets import load_dataset

    ds = load_dataset("beans", name="full")

    ds_train = ds["train"]
    ds_validation = ds["validation"]

    train_dataset = ds_train.with_transform(train_batch_transforms)
    validation_dataset = ds_validation.with_transform(validation_batch_transforms)

    return train_dataset, validation_dataset
