import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# self defined dataset
from clip_align.dataset import EmbeddingDataset
from clip_align.converter import Converter, Converter_Att, Converter_Linear
from clip_align.loss import AlignLoss
from clip_align.vis import visualize_projection, visualize_similarity