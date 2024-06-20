import conllu
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def preprocess_conllu(
    file: str, tokenizer: RobertaTokenizerFast, model: RobertaModel
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess a conllu file into a tensor
    @param file: path to the conllu file
    @return: tuple of preprocessed input tensor and label tensor
    """

    device = deviceof(model)

    embedded_text_tensors = []

    # read file
    with open(file, "r") as f:
        for i, token_list in enumerate(conllu.parse_incr(f)):
            print(i)
            # get conllu spans of the token list
            token_spans = get_token_spans(token_list)
            # get the overall string
            ex_string = " ".join(t["form"] for t in token_list)
            # tokenize
            inputs = tokenizer(ex_string, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # get the hidden states
            last_hidden_state = model(input_ids, attention_mask).last_hidden_state[0]

            # merge hidden states for each conllu token
            token_embs = []
            for t, (start, end) in zip(token_list, token_spans):
                start = inputs.char_to_token(0, start)
                end = inputs.char_to_token(0, end)
                token_embs.append(last_hidden_state[start : end + 1].mean(dim=0))

            embedded_text_tensors.append(torch.stack(token_embs))

    max_length = max(t.shape[0] for t in embedded_text_tensors)

    return (
        torch.stack(
            [F.pad(t, (0, 0, 0, max_length - len(t))) for t in embedded_text_tensors]
        ),
        torch.tensor([len(t) for t in embedded_text_tensors]),
    )


class TransitionParser:
    def __init__(self, model: ParseOracle):
        self.model = model

    def train(self, data: list[conllu.TokenTree]):
        pass
