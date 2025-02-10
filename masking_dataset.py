from torch.utils.data import Dataset
import torch
import random
import itertools


def flatten(l):
    return itertools.chain.from_iterable(l)


class MaskingDataset(Dataset):
    def __init__(self, samples, tokenizer, sample_length, replacement_prob=0.15, mask_prob=0.8, random_token_prob=0.1, special_tokens=None):
        super().__init__()

        assert replacement_prob < 1.0
        assert mask_prob <= 1.0
        assert random_token_prob <= 1.0

        self.samples = samples
        self.tokenizer = tokenizer
        self.sample_length = sample_length
        self.replacement_prob = replacement_prob
        self.vocab = list(set(tokenizer.get_vocab().values()) - set(tokenizer.all_special_ids) - set(special_tokens or {}))
        # Accumulate probs
        self.mask_prob = mask_prob
        self.random_token_prob = random_token_prob + self.mask_prob
        self.unchanged_prob = 1.0 - mask_prob + random_token_prob

    @staticmethod
    def merge(encodings):
        return (list(flatten(ee)) for ee in zip(*((e.ids, e.attention_mask, e.special_tokens_mask) for e in encodings)))

    def build_tokenized(self, sentences):
        tokens = self.tokenizer(sentences[0])

        first_encoding = tokens.encodings[0]
        num_tokens = len(first_encoding)
        if num_tokens > self.sample_length:
            print("Bad batch - tokenized first sentence exceeded sample length")
            #assert num_tokens <= self.sample_length, "Bad batch - tokenized first sentence exceeded sample length"

        input_ids = torch.full((self.sample_length,), self.tokenizer.pad_token_id)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        special_tokens_mask = torch.ones_like(input_ids)

        # Fill sample as much as possible with sentences and use a <sep> token between.
        mencodings = [first_encoding]
        for sentence in sentences[1:]:
            next_tokens = self.tokenizer(sentence)
            next_encoding = next_tokens.encodings[0]
            next_encoding.truncate(len(next_encoding)-1, direction='left')
            next_num_tokens = len(next_encoding)
            if num_tokens + next_num_tokens > self.sample_length:
                break
            mencodings.append(next_encoding)
            num_tokens += next_num_tokens

        ids, attn, spct = MaskingDataset.merge(mencodings)
        me_len = min(len(ids), self.sample_length)
        input_ids[:me_len] = torch.tensor(ids[:me_len])
        attention_mask[:me_len] = torch.tensor(attn[:me_len], dtype=torch.bool)
        special_tokens_mask[:me_len] = torch.tensor(spct[:me_len])

        return input_ids, attention_mask.outer(attention_mask), special_tokens_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sentences = self.samples[idx]
        input_ids, attention_mask, special_tokens_mask = self.build_tokenized(sentences)
        # Generate a random sample of indices without replacement without hitting special tokens
        weights = special_tokens_mask
        weights = weights.logical_not()
        num_tokens = weights.sum()

        # Mask a percentage or the words.
        num_indices = num_tokens * self.replacement_prob
        num_indices = num_indices.int()
        weights = weights.float()

        replacement_indices = torch.multinomial(weights, num_indices) if num_indices > 0 else []
        replaced_token_ids = input_ids.detach().clone()

        for replacement_idx in replacement_indices:
            r = random.random()
            if r < self.mask_prob:
                replaced_token_ids[replacement_idx] = self.tokenizer.mask_token_id
                continue
            if r < self.random_token_prob:
                replaced_token_ids[replacement_idx] = random.choice(self.vocab)
                continue

        return replaced_token_ids, attention_mask, input_ids
