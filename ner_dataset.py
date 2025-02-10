from torch.utils.data import Dataset
import torch
from tokenizers import Encoding
import random

from transformers import PreTrainedTokenizerBase


class NerDataset(Dataset):
    def __init__(self, docs, labels, tokenizer: PreTrainedTokenizerBase, sample_length, replacement_prob=0.15, mask_prob=0.8, random_token_prob=0.1, special_tokens=None):
        super().__init__()

        assert replacement_prob < 1.0
        assert mask_prob <= 1.0
        assert random_token_prob <= 1.0

        self.docs = docs
        self.labels = labels
        self.tokenizer = tokenizer
        self.sample_length = sample_length
        self.replacement_prob = replacement_prob
        self.vocab = list(set(tokenizer.get_vocab().values()) - set(tokenizer.all_special_ids) - set(special_tokens or {}))
        # Accumulate probs
        self.mask_prob = mask_prob
        self.random_token_prob = random_token_prob + self.mask_prob
        self.unchanged_prob = 1.0 - mask_prob + random_token_prob

    @staticmethod
    def subword_level_alignment(offset_mapping):
        token_idx = []
        count = 0
        prev = 0
        for om in offset_mapping:
            if om == (0,0):  # Empty token? Special token?
                token_idx.append(-1)
                continue
            start, nprev = om
            # There is a space between prev and current token, if there is an offset of 1
            count += int(start == prev+1)
            token_idx.append(count)
            prev = nprev
        return token_idx

    def build_tokenized(self, sentences, labels):
        tokens = self.tokenizer(sentences[0], return_offsets_mapping=True)

        first_encoding = tokens.encodings[0]
        num_tokens = len(first_encoding)
        assert num_tokens <= self.sample_length, "Bad batch - tokenized first sentence exceeded sample length"
        alignment = NerDataset.subword_level_alignment(tokens['offset_mapping'])
        mlabels = [labels[0][i] if i >= 0 else 0 for i in alignment]

        # Fill sample as much as possible with sentences and use a <sep> token between.
        mencodings = [first_encoding]
        for sentence, label in zip(sentences[1:], labels[1:]):
            next_tokens = self.tokenizer(sentence, return_offsets_mapping=True)
            next_encoding = next_tokens.encodings[0]
            next_num_tokens = len(next_encoding)
            next_encoding.truncate(next_num_tokens-1, direction='left')
            if num_tokens + next_num_tokens > self.sample_length:
                break
            mencodings.append(next_encoding)
            alignment = NerDataset.subword_level_alignment(next_tokens['offset_mapping'][1:])
            mlabels += [label[i] if i >= 0 else 0 for i in alignment]
            num_tokens += next_num_tokens

        me = Encoding.merge(mencodings)
        me_len = len(me)
        input_ids = torch.full((self.sample_length,), self.tokenizer.pad_token_id)
        input_labels = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        special_tokens_mask = torch.ones_like(input_ids)

        input_ids[:me_len] = torch.tensor(me.ids)
        input_labels[:me_len] = torch.tensor(mlabels)
        attention_mask[:me_len] = torch.tensor(me.attention_mask)
        special_tokens_mask[:me_len] = torch.tensor(me.special_tokens_mask)

        return input_ids, input_labels, attention_mask

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):

        sentences = self.docs[idx]  # shape (B, X)
        labels = self.labels[idx]

        return self.build_tokenized(sentences, labels)