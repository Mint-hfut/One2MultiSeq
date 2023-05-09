import os.path
from transformers import BartConfig,BartTokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right
import torch
model_name = 'facebook/bart-base'
config = BartConfig()
tokenizer = BartTokenizer.from_pretrained(model_name, sep_token='<sep>')
tokenizer.add_tokens(['<number>', '<url>', '<mention>'], special_tokens=True)
def encode(src,trg,encoder_input_length,decoder_input_length, is_CMKP_data):
    original_text_encodings = None
    if is_CMKP_data:
        original_text = src.split('<seg>')[0]
        original_text_encodings = tokenizer(original_text, max_length=encoder_input_length, pad_to_max_length=True, return_tensors="pt",
                                truncation=True)
        src = src.replace('<seg>', ' and ')
        input_encodings = tokenizer(src, max_length=encoder_input_length, pad_to_max_length=True, return_tensors="pt",
                                    truncation=True)
    else:
        input_encodings = tokenizer(src, max_length=encoder_input_length, pad_to_max_length=True, return_tensors="pt",
                                    truncation=True)
    input_ids = input_encodings['input_ids']
    attention_mask = input_encodings['attention_mask']
    if original_text_encodings is not None:
        original_text_attention_mask = original_text_encodings['attention_mask']
    else:
        original_text_attention_mask = input_encodings['attention_mask'].clone()
    target_encodings = tokenizer(trg, max_length=decoder_input_length, pad_to_max_length=True, return_tensors="pt",
                                 truncation=True)
    labels = target_encodings['input_ids']
    decoder_input_ids = shift_tokens_right(labels, config.pad_token_id,
                                           config.decoder_start_token_id)
    labels[labels[:, :] == config.pad_token_id] = -100

    encodings = {
        'input_ids': input_ids.squeeze(),
        'attention_mask': attention_mask.squeeze(),
        'labels': labels.squeeze(),
        'decoder_input_ids': decoder_input_ids.squeeze(),
        'use_cache_original_text_attention_mask': original_text_attention_mask.squeeze(),
    }
    return encodings

def dataprocess(src_path,data_type,encoder_input_length,decoder_input_length,is_CMKP_data):
    all_examples = []
    with open(src_path+f'/{data_type}_src.txt', 'r', encoding='utf-8') as src_lines:
        with open(src_path+f'/{data_type}_trg.txt', 'r', encoding='utf-8') as trg_lines:
            if data_type =='train':
                is_train = True
                paradigm = 'MultiSeq'
            else:
                is_train = False
                paradigm = 'Seq'
            src_line = src_lines.readlines()
            trg_line = trg_lines.readlines()
            assert len(src_line) == len(trg_line)
            if os.path.exists(src_path+f'/{data_type}_One2{paradigm}.pt'):
                return
            for i in range(len(src_line)):
                src = src_line[i].strip()
                trg = trg_line[i].strip()
                if len(src)<2:
                    continue
                trgs = trg.split(';')
                if '<peos>' in trgs:
                    trgs.remove('<peos>')
                elif '<peos>\n' in trgs:
                    trgs.remove('<peos>\n')
                else:
                    pass
                if src == '\n':
                    continue
                if data_type != 'test':
                    present = []
                    absent = []
                    for i in trgs:
                        if i in src:
                            present.append(i)
                        else:
                            absent.append(i)
                    trgs_ordered = present + absent
                else:
                    trgs_ordered = trgs
                trgs_connect1 = ['<sep>'.join(k.strip() for k in trgs_ordered)]
                encodings = encode(src,trgs_connect1,encoder_input_length,decoder_input_length, is_CMKP_data)

                all_examples.append(encodings)
                if is_train:
                    trgs_ordered.reverse()
                    trgs_connect2 = ['<sep>'.join(k.strip() for k in trgs_ordered)]
                    encodings = encode(src, trgs_connect2,encoder_input_length,decoder_input_length,is_CMKP_data)
                    all_examples.append(encodings)
                if len(all_examples) % 10000 == 0:
                    print(f"completed：{len(all_examples)}",)
    torch.save(all_examples,open(src_path+f'/{data_type}_One2{paradigm}.pt', 'wb'))


datapath = [[['data/CMKP_data'],['train','valid','test']],[['data/Twitter_data'],['train','valid','test']],[['data/StackExchange_data'],['train','valid','test']]
            ,[['data/KP20K/kp20k_separated'],['train','valid']],[['data/KP20K/testsets/kp20k'],['test']],[['data/KP20K/testsets/inspec'],['test']],[['data/KP20K/testsets/krapivin'],['test']],
            [['data/KP20K/testsets/nus'],['test']],[['data/KP20K/testsets/semeval'],['test']]]

for path, datatypes in datapath:
    if 'kp20k_separated' in path[0]:
        tokenizer.add_tokens(['<eos>'], special_tokens=True)
    for datatype in datatypes:
            if 'CMKP_data' in path[0]:
                encoder_input_length = 125
                decoder_input_length = 16
                is_CMKP_data = True
            elif 'Twitter_data' in path[0]:
                encoder_input_length = 64
                decoder_input_length = 16
                is_CMKP_data = False
            elif 'StackExchange_data' in path[0]:
                encoder_input_length = 128
                decoder_input_length = 24
                is_CMKP_data = False
            else:
                encoder_input_length = 192
                decoder_input_length = 96
                is_CMKP_data = False
            dataprocess(path[0], datatype, encoder_input_length, decoder_input_length,is_CMKP_data)