import numpy as np
import pandas as pd
from scipy.stats import zscore
from datasets import load_dataset
from datasets import Dataset

from NeuroStrikeProject.white_box import util

subset = util.load_sorted_datasets(False, "hate", True)
print(subset)