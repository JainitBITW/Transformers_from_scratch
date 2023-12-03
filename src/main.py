import torch 
from torch import nn
import torch.nn.functional as F
import math
import copy
import re 
import pytorch_lightning as pl
from torch.utils.data import DataLoader , Dataset
from tokeniser import *
from Dataset import *
from archi import *
from config_t import *

if checkpoint : 
    model = LitTransformer.load_from_checkpoint(checkpoint)
else :
    
    model = LitTransformer(d_model , num_heads , d_ff , num_layers , dropout , en_vocab , fr_vocab )

trainer = pl.Trainer(max_epochs=epochs  , enable_progress_bar=True, logger=None ,devices =1 ) 


trainer.fit(model , LitDataset(dataloaders))

