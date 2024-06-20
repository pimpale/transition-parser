
import conllu
import torch
from transformers import RobertaTokenizerFast, RobertaModel
import numpy as np

from parse_oracle import ParseOracle

def deviceof(model: torch.nn.Module) -> torch.device:
    """
    Get the device of a model
    @param model: pytorch model
    @return: device of the model
    """
    return next(model.parameters()).device

def get_token_spans(tokens: conllu.TokenList) -> list[tuple[int, int]]:
    """
    Get the start and end indices of the tokens in a sentence
    @param tokens: list of tokens
    @return: list of tuples of start and end indices
    """
    word_start_idxs = []
    word_end_idxs = []
    word_start = 0
    for token in tokens:
        word_start_idxs.append(word_start)
        word_len = len(token["form"])
        word_end_idxs.append(word_start + word_len - 1)
        word_start += len(token["form"]) + 1
    return list(zip(word_start_idxs, word_end_idxs))

def preprocess_conllu(file: str, tokenizer: RobertaTokenizerFast, model: RobertaModel) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess a conllu file into a tensor
    @param file: path to the conllu file
    @return: tuple of preprocessed input tensor and label tensor
    """
    

    
    device = deviceof(model)
    
    embedded_text_tensors = []
    
    # read file
    with open(file, "r") as f:        
        for token_list in conllu.parse_incr(f):
            # get conllu spans of the token list
            token_spans = get_token_spans(token_list)
            # get the overall string
            ex_string = ' '.join(t["form"] for t in token_list)
            # tokenize
            inputs = tokenizer(ex_string, return_tensors="pt")
            input_ids = inputs["input_ids"][0].to(device)
            attention_mask = inputs["attention_mask"][0].to(device)
            
            # get the hidden states
            last_hidden_state = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0)).last_hidden_state[0]
            
            # merge hidden states for each conllu token
            token_embs = []
            for t, (start, end) in zip(token_list, token_spans):
                print(t['form'])
                print('c', start, end)
                start = inputs.char_to_token(start)
                end = inputs.char_to_token(end)+1
                print('t', start, end)
                print(input_ids[start:end])
                token_embs.append(last_hidden_state[start:end].mean(dim=0))
            
            embedded_text_tensors.append(torch.cat(token_embs))
        
    return torch.stack(embedded_text_tensors)
    

class TransitionParser:
    def __init__(self, model: ParseOracle):
        self.model = model
    
    
    def train(self, data: list[conllu.TokenTree]):
        pass
    