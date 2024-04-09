import torch
import torch.nn as nn

from torch.utils.data import Dataset


class PositiveDataset(Dataset):
  def __init__(self, source, target, tokenizer, config):
    super().__init__()

    self.txt = source.values
    self.target = target.values

    self.seq_len = config['seq_len']
    self.h = config['h']
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.txt)

  def __getitem__(self, idx):
    source = str(self.txt[idx])
    target = str(self.target[idx])

    h = self.h
    seq_len = self.seq_len

    # tokenized_source
    tokenized_source = self.tokenizer(source,
                                     max_length = self.seq_len,
                                     padding='max_length',
                                     truncation = True
                                     )

    # encoder inputs
    encoder_input = torch.tensor(tokenized_source['input_ids'], dtype= torch.long)

    # source masks
    encoder_mask = torch.tensor(tokenized_source['attention_mask'], dtype= torch.long).unsqueeze(0)
    encoder_mask = encoder_mask.repeat(1, h, 1)
    encoder_mask = encoder_mask.expand(seq_len, h, seq_len)
    encoder_mask = encoder_mask.transpose(0,1).contiguous()

    # tokenized_target
    tokenized_target = self.tokenizer(target,
                                     max_length = self.seq_len,
                                     padding='max_length',
                                     truncation = True
                                     )

    # decoder inputs (should not be used in loss function. It's an input)
    decoder_input = torch.tensor(tokenized_target['input_ids'], dtype= torch.long)
    # remove special tokens for end of statement
    decoder_input = torch.where(decoder_input == 102, torch.tensor(0), decoder_input)

    # target_masks
    decoder_mask = torch.tensor(tokenized_target['attention_mask'], dtype= torch.long).unsqueeze(0)
    decoder_mask = decoder_mask.repeat(1, h, 1)
    decoder_mask = decoder_mask.expand(seq_len, h, seq_len)
    decoder_mask = decoder_mask.transpose(0,1).contiguous().type(torch.int)
    # target_masks must be causally masked
    decoder_mask = decoder_mask & torch.tril(torch.ones(h, seq_len, seq_len)).type(torch.int)

    # label (should be used in loss function)
    label = torch.tensor(tokenized_target['input_ids'], dtype= torch.long)
    # remove special tokens for start of statement
    label = torch.cat((label[1:], torch.tensor([0])))


    item = {
          'source_txt': source,
          'target_txt': target,
          'encoder_input': encoder_input,
          'encoder_mask': encoder_mask,
          'decoder_input': decoder_input,
          'decoder_mask': decoder_mask,
          'label' : label
    }

    return item
