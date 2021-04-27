#!pip install torchtext==0.4.0

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchtext.vocab as vocab
from DataPreprocess import *
from TrainEvalLoops import *
from model import *

choice = input("Type 'h' to run model for Hindi and 'b' for Bengali:\n")
if choice == 'h':
    print(f'You entered {choice}, running sentiment analysis for Hindi.')

    #HINDI
    url = 'Data/hindi_hatespeech.tsv'
    df = pd.read_csv(url, sep='\t')
    #clean data
    df['text'] = clean_data(df['text'])
    df['text'] = remove_hindi_stopwords(df['text'])
    df = hindi_drop_columns(df) #dropping columns we don't need 
    #load our custom embeddings into vocab vector
    #custom_embeddings = vocab.Vectors(name = 'HindiEmbeddings.txt') #basic skipgram embeddings
    custom_embeddings = vocab.Vectors(name = 'Data/HindiEmbeddingsUpdated.txt') #sgns embeddings

if choice == 'b':
    print(f'You entered {choice}, running sentiment analysis for Bengali.')

    #BENGALI
    url = 'Data/bengali_hatespeech.csv'
    df = pd.read_csv(url,header = None)
    df = reduce_bengali(df)
    #clean data
    df[0] = clean_data(df[0])
    df[0] = remove_bengali_stopwords(df[0])
    df = bengali_drop_columns(df) #dropping columns we don't need 
    #load our custom embeddings into vocab vector
    #custom_embeddings = vocab.Vectors(name = 'Data/BengaliEmb.txt') #basic skipgram embeddings
    custom_embeddings = vocab.Vectors(name = 'Data/BengaliEmbeddingsUpdated.txt') #sgns embeddings

TEXT = data.Field(sequential=True, batch_first=True)
LABEL = data.LabelField(dtype=torch.float)

df.columns = ['text','label'] 
fields = { 'text' : TEXT,  'label':LABEL  }
train_ds = DataFrameDataset(df, fields) #convert pandas df to torchtext df

TEXT.build_vocab(train_ds, vectors = custom_embeddings)
LABEL.build_vocab(train_ds)
#store our pretrained word embeddings in this variable to pass to our network
pretrained = TEXT.vocab.vectors

#make test and train splits
SEED = 32
train_size = int(0.75*len(train_ds)) #use 75% for training and rest for testing
val_size = len(train_ds)-train_size
train_dataset,val_dataset = train_ds.split([train_size,val_size], random_state = random.seed(SEED))

#define hyperparamters
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
N_FILTERS = 200
FILTER_SIZES = [2,3,4,5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
BATCH_SIZE = 500
EPOCHS = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#create our dataloaders
train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_dataset, val_dataset), 
    batch_size = BATCH_SIZE,
    sort_within_batch = False,
    sort_key = lambda x: len(x.text),
    device = device)


model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

#ensure embedding in hidden layer for unknown and padding are set to zero
model.embedding.weight.data.copy_(pretrained)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

optimizer = optim.Adam(model.parameters(),  lr=5e-3)

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)

criterion = criterion.to(device)

for epoch in range(EPOCHS):

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')



test_loss, test_acc = evaluate(model, valid_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
