import pytorch_lightning as pl
from torch.utils.data import DataLoader , Dataset

from tokeniser import * 

class TranslationDataset(Dataset):
    def __init__(self , en_input_ids , en_attention_mask , fr_input_ids , fr_attention_mask):
        self.en_input_ids = en_input_ids
        self.en_attention_mask = en_attention_mask
        self.fr_input_ids = fr_input_ids
        self.fr_attention_mask = fr_attention_mask
    
    def __len__(self):
        return len(self.en_input_ids)
    
    def __getitem__(self , idx):
        return ((self.en_input_ids[idx] , self.en_attention_mask[idx]) , (self.fr_input_ids[idx] , self.fr_attention_mask[idx]))
    
class LitDataset(pl.LightningDataModule):
    def __init__(self , dataloaders):
        super().__init__()
        self.dataloaders = dataloaders
    
    def train_dataloader(self):
        return self.dataloaders["train"]
    
    def val_dataloader(self):
        return self.dataloaders["dev"]
    
    def test_dataloader(self):
        return self.dataloaders["test"]
    
data = {"train":{"en":[],"fr":[]},"test":{"en":[],"fr":[]},"dev":{"en":[],"fr":[]}}
for type in ["train" , "test" , "dev"]:
    for lang in ["en" , "fr"]:
        with open(f"../data/{type}.{lang}" , "r") as f:
            for line in f:
                data[type][lang].append(line.strip())
                
            
english_corpus = data["train"]["en"] + data["dev"]["en"]
french_corpus = data["train"]["fr"] + data["dev"]["fr"]

en_word2idx , en_idx2word , en_vocab = preprocess_corpus(english_corpus, threshold = 5)
fr_word2idx , fr_idx2word , fr_vocab = preprocess_corpus(french_corpus , threshold = 5)


en_tokeniser = Tokeniser(en_word2idx , en_idx2word)
fr_tokeniser = Tokeniser(fr_word2idx , fr_idx2word)

en_input_ids , en_attention_mask = en_tokeniser(data["train"]["en"])
fr_input_ids , fr_attention_mask = fr_tokeniser(data["train"]["fr"])

train_dataset = TranslationDataset(en_input_ids , en_attention_mask , fr_input_ids , fr_attention_mask)
en_input_ids_dev, en_attention_mask_dev = en_tokeniser(data["dev"]["en"])
fr_input_ids_dev, fr_attention_mask_dev = fr_tokeniser(data["dev"]["fr"])
dev_dataset = TranslationDataset(en_input_ids_dev , en_attention_mask_dev , fr_input_ids_dev , fr_attention_mask_dev)
en_input_ids_test, en_attention_mask_test = en_tokeniser(data["test"]["en"])
fr_input_ids_test, fr_attention_mask_test = fr_tokeniser(data["test"]["fr"])
test_dataset = TranslationDataset(en_input_ids_test , en_attention_mask_test , fr_input_ids_test , fr_attention_mask_test)
dataloaders = {"train":DataLoader(train_dataset , batch_size = 32 , shuffle = True , num_workers = 4) ,
                "dev":DataLoader(dev_dataset , batch_size = 32 , shuffle = False , num_workers = 4) ,
                "test":DataLoader(test_dataset , batch_size = 32 , shuffle = False , num_workers = 4)}

