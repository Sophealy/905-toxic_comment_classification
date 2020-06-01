#!/usr/bin/env python

import click
import torch

# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


import pandas as pd

from sklearn.metrics import roc_auc_score

# Team Modules
from dataloader import ToxicDataset
from dl_models import BiLSTM,LSTMAttention,CNN
from vectorizer import RobertaVectorizer,XLNetVectorizer


# Training porameters
EPOCHS =4
BATCH_SIZE = 100


# Word embeddings
roberta  = RobertaVectorizer()
xlnet = XLNetVectorizer()

#DNN
bilstm = BiLSTM(output_size = 6,hidden_size = 512,embedding_length = 768 )
lstmatt = LSTMAttention(output_size = 6,hidden_size = 512, embedding_length = 768 )
cnn  = CNN(output_size = 6,kernel_heights = [5,3,2],embedding_length = 768)

CONFIG_MODEL = {

    "robertalstm":(roberta,bilstm),
    "robertalstmatt":(roberta,lstmatt),
    "robertacnn":(roberta,cnn),
    "xlnetlstm":(roberta,bilstm),
    "xlnetlstmatt":(roberta,lstmatt),
    "xlnetcnn":(roberta,cnn),
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loading_resources(path_toxic, model):
    data  = ToxicDataset(path_toxic)
    embeddings, model = CONFIG_MODEL[model]
    embeddings.load_model()
    model = model.to(DEVICE)
    training_data = DataLoader(data,batch_size = BATCH_SIZE, pin_memory=True)

    return training_data,embeddings,model


@click.group()
def cli():
    pass

@click.option('--dir',default ="data/processed/embeddings/" )
@click.option('--test-data',default = "data/processed/test.csv")
@click.option('--train-data',default = "data/processed/train.csv")
@cli.command()
def embed(train_data,test_data,dir):


    print("Loading vectorizer models")
    roberta.load_model()
    xlnet.load_model()

    def vectorize_comment(c):
        comment = c.comment_text
        comment_id = c.id


        for vectorizer in [(roberta,"roberta"),(xlnet,"xlnet")]:
            comment_vec = vectorizer[0].transform(comment)
            torch.save(comment_vec, dir + '{}-{}.pt'.format(vectorizer[1],comment_id))

    
    for data in [test_data]:
        df = pd.read_csv(data)
        len_df = len(df)
        i = 0
        for row in df.itertuples(index=False):
            if i % 50 == 0:
                print("{}/{}".format(i,len_df))
            i = i +1
            vectorize_comment(row)

       




@click.option('--dir',default ="data/processed/embeddings/" )
@click.option('--epochs', default=EPOCHS)
@click.option('--model-name', default = "xlnetcnn")
@click.option('--lr',default = 0.1)
@click.option('--td',default = "data/processed/train.csv")
@cli.command()
def train(dir,epochs,model_name,lr,td):


    training_data,embeddings,model = loading_resources(td,model_name)


    click.echo('Begin of the training phase')

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)


    # Trainning loop
    i = 1
    for epoch in range(epochs):
        for comments, tags in  training_data:
            
            if  "roberta" in model_name:
                name = "roberta-"
            else:
                name="xlnet-"

            # Get our inputs ready for the network,
            comments_vec =pad_sequence([torch.load(dir+name+id+".pt") for id in comments])

            # We need to clear them out before each instance
            model.zero_grad()

            # Feed the model
            tag_scores = model(comments_vec)

            
            # Compute the loss, gradients, and update the parameters by
            #calling optimizer.step()

            click.echo("Epoch {}, batch number {}".format(epoch+1,i*BATCH_SIZE))
            loss = loss_function(tag_scores, tags.to(DEVICE))
            print("Loss: ",loss)

            loss.backward()
            optimizer.step()

            i+=1

    torch.save(model,"config/{}".format(model_name) )
    click.echo("Learning process done, exporting the model")

@cli.command()
@click.option('--dir-name',default ="data/processed/embeddings/" )
@click.option('--model-name', default = "xlnetcnn")
@click.option('--ed',default = "data/processed/test.csv")
def eval(dir_name,model_name,ed):

    click.echo('Begin of the test phase')

    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    with torch.no_grad():

    
        model = torch.load("config/{}".format(model_name))
        model.eval()

        test_data,embeddings,_ = loading_resources(ed,model_name)

        if  "roberta" in model_name:
            name = "roberta-"
        else:
            name="xlnet-"


        y,y_pred = [], []

        for comments, tags in test_data:
            embeds=pad_sequence([torch.load(dir_name+name+c+".pt") for c in comments])


            y.append(tags)
            y_pred.append(model(embeds))

        y = torch.cat(y)
        y_pred = torch.cat(y_pred)
        
        print(y.shape)
        print(y_pred.shape)
        print(y)
        
        loss_function = torch.nn.BCEWithLogitsLoss()

        loss = loss_function(y_pred, y.to(DEVICE))
        print("Loss: ",loss)

        


if __name__ == '__main__':
    cli()
