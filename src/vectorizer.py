from transformers import *
import torch


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model ,tokenizer, pretrained_weights):

    model =  model.from_pretrained(pretrained_weights).to(DEVICE)
    tokenizer =  tokenizer.from_pretrained(pretrained_weights)

    return model,tokenizer

def sent_transform(sent,max_position_embeddings,tokenizer,model):

    
    input_ids = [tokenizer.encode(sent, add_special_tokens=True)]
    if len(input_ids[0]) >= max_position_embeddings:
        input_ids = [input_ids[0][:max_position_embeddings]]

    with torch.no_grad():
        last_state = model(torch.tensor(input_ids).to(DEVICE))[0]
        return torch.squeeze(last_state)
    
    
    
    
     
class RobertaVectorizer():
    def load_model(self,pretrained_weigths = "roberta-base"):
        self.model,self.tokenizer = load_model(RobertaModel,RobertaTokenizer,pretrained_weigths)
    def transform(self,sent,max_position_embeddings = 512):
        return sent_transform(sent, max_position_embeddings,self.tokenizer,self.model)

class XLNetVectorizer():
    def load_model(self,pretrained_weigths = "xlnet-base-cased"):
        self.model,self.tokenizer = load_model(XLNetModel,XLNetTokenizer,pretrained_weigths)
    def transform(self,sent,max_position_embeddings = 512):
        return sent_transform(sent, max_position_embeddings,self.tokenizer,self.model)
