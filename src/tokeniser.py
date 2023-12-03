import torch 
import re 


def preprocess_sentence(sentence):
    """This function preprocesses the sentence"""
    #lowercase the sentence
    sentence = sentence.lower()
    #split the sentence into word
    puntuations = r" !\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
    sentence = re.sub(puntuations, r" \1 ", sentence)
    #replace continous numbers wiith <num>
    sentence = re.sub(r"\d+", "<num>", sentence)
    #replace multiple spaces with single space
    sentence = re.sub(r" +", " ", sentence)
    #split the sentence into words
    return sentence 
    

def preprocess_corpus(sentences , threshold = 5):
    """This function is for all sentences """
    
    word_freq = {}
    # first preprocess the sentences 
    tokenised_sentences = []
    for i , sentence in enumerate(sentences):
        #lowercase the sentence
        sentence = preprocess_sentence(sentence)
        #split the sentence into words
        words = sentence.split()
        tokenised_sentences.append(words)
        #count the frequency of each word
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    
    #remove the words with frequency less than threshold
    #replace the words with frequency less than threshold with <unk>
    vocab = ["<pad>" , "<sos>" , "<eos>" , "<unk>"]
    for i , sentence in enumerate(tokenised_sentences):
        for j , word in enumerate(sentence):
            if word_freq[word] < threshold:
                tokenised_sentences[i][j] = "<unk>"
    
    #create the vocab
    for word in word_freq:
        if word_freq[word] >= threshold:
            vocab.append(word)
            
    #create the word2idx and idx2word
    word2idx = {}
    idx2word = {}
    for i , word in enumerate(vocab):
        word2idx[word] = i
        idx2word[i] = word
    
    return  word2idx , idx2word , vocab 



# get the train and test sentences with labels 
data = {"train":{"en":[],"fr":[]},"test":{"en":[],"fr":[]},"dev":{"en":[],"fr":[]}}
for type in ["train" , "test" , "dev"]:
    for lang in ["en" , "fr"]:
        with open(f"../data/{type}.{lang}" , "r") as f:
            for line in f:
                data[type][lang].append(line.strip())
                
                

class Tokeniser():
    def __init__(self , word2idx , idx2word):
        self.word2idx = word2idx
        self.idx2word = idx2word
    
    def __call__(self , sentences, truncation = 100 , padding = True):
        """This function tokenises the sentence"""
        #first preprocess the sentences
        max_len = 0
        tokenised_sentences = []
        for sentence in sentences:
            #lowercase the sentence
            sentence = preprocess_sentence(sentence)
            #split the sentence into words
            words = sentence.split()
            tokenised_sentences.append(words)
            if len(tokenised_sentences[-1] ) > truncation-2:
                tokenised_sentences[-1] = tokenised_sentences[-1][:truncation]
                tokenised_sentences[-1]= ["<sos>"] + tokenised_sentences[-1] + ["<eos>"]
            else :
                tokenised_sentences[-1]= ["<sos>"] + tokenised_sentences[-1] + ["<eos>"]
            if len(tokenised_sentences[-1]) > max_len:
                max_len = len(tokenised_sentences[-1])
            
        
        #convert the words into indices
        indices, attention_mask = [] , []
        for sentence in tokenised_sentences:
            index = []
            for word in sentence:
                if word in self.word2idx:
                    index.append(self.word2idx[word])
                else:
                    index.append(self.word2idx["<unk>"])
            #pad the sentences
            
            if padding:
                
                index = index + [self.word2idx["<pad>"]] * (max_len - len(index))
            index = torch.tensor(index)
            indices.append(index)
        
        
        #create the attention mask
        for index in indices:
            attention_mask.append(torch.where(index == self.word2idx["<pad>"] , 0 , 1))
        
        #convert the indices into tensor
        indices = torch.stack(indices)
        attention_mask = torch.stack(attention_mask)
        return indices , attention_mask
        
        
    def decode(self , indices):
        """This function decodes the indices into words"""
        words = []
        for index in indices:
            words.append(self.idx2word[index])
        return words