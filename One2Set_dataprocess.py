from transformers import BartConfig,BartTokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right
import torch
import os
model_name = 'facebook/bart-base'
encoder_input_length=192
decoder_input_length=12
config = BartConfig()
tokenizer = BartTokenizer.from_pretrained(model_name)
tokenizer.add_tokens(['<number>', '<url>', '<mention>', '<digit>', '<eos>', '<NULL>'], special_tokens=True)
max_kp_num = 20
def dataprocess(src_path, data_type):
    with open(src_path+f'/{data_type}_src.txt', 'r', encoding='utf-8') as src_lines:
        with open(src_path+f'/{data_type}_trg.txt', 'r', encoding='utf-8') as trg_lines:
            all_examples = []
            remove_title_eos=True
            srcs = src_lines.readlines()
            trgs = trg_lines.readlines()
            # if os.path.exists(src_path+f'/{data_type}_One2Set.pt'):
            #     return
            for i in range(len(srcs)):
                if len(srcs[i])<=2:
                    continue
                src = srcs[i].strip()
                title_and_context = src.split('<eos>')
                if len(title_and_context) == 1:  # it only has context without title
                    [context] = title_and_context
                    src_word_list = context.strip().split(' ')
                elif len(title_and_context) == 2:
                    [title, context] = title_and_context
                    title_word_list = title.strip().split(' ')
                    context_word_list = context.strip().split(' ')
                    if remove_title_eos:
                        src_word_list = title_word_list + context_word_list
                    else:
                        src_word_list = title_word_list + ['<eos>'] + context_word_list
                else:
                    raise ValueError("The source text contains more than one title")
                trg = trgs[i].split(';')
                preset_trg = []
                absent_trg = []
                if 'KP20K' in src_path:
                    present_is_end = False
                    for word in trg:
                        if '<peos' not in word and present_is_end==False:
                            preset_trg.append(word.strip())
                        elif '<peos>' == word.strip():
                            present_is_end = True
                            continue
                        else:
                            absent_trg.append(word.strip())
                else:
                    for word in trg:
                        if word.strip() in src:
                            preset_trg.append(word.strip())
                        else:
                            absent_trg.append(word.strip())
                mid_len = max_kp_num // 2
                if len(preset_trg) > mid_len:
                    preset_trg = preset_trg[:mid_len]
                if len(absent_trg) > mid_len:
                    absent_trg = absent_trg[:mid_len]
                extra_present_targets = ['<NULL>'] * (mid_len- len(preset_trg))
                extra_absent_targets = ['<NULL>'] * (mid_len - len(absent_trg))
                all_trgs = preset_trg + extra_present_targets + absent_trg+extra_absent_targets
                if len(all_trgs)>20:
                    print('-------------------')
                input_encodings = tokenizer(src, max_length=encoder_input_length, pad_to_max_length=True, return_tensors="pt",
                                         truncation=True)
                input_ids = input_encodings['input_ids']
                attention_mask = input_encodings['attention_mask']

                target_encodings = tokenizer(all_trgs, max_length=decoder_input_length, pad_to_max_length=True,
                                             return_tensors="pt",
                                             truncation=True)
                trg_mask = target_encodings['attention_mask']
                labels = target_encodings['input_ids']
                decoder_input_ids = shift_tokens_right(labels, config.pad_token_id,
                                                       config.decoder_start_token_id)
                encodings = {
                    'input_ids': input_ids.squeeze(),
                    'attention_mask': attention_mask.squeeze(),
                    'labels': labels.squeeze(),
                    'trg_mask':trg_mask.squeeze(),
                    'decoder_input_ids': decoder_input_ids.squeeze(),

                    'src_word_list':src_word_list,
                }
                all_examples.append(encodings)
                if len(all_examples) % 10000 == 0:

                    print(f"completed：{len(all_examples)}",)
    torch.save(all_examples,open(src_path+f'/{data_type}_One2Set.pt', 'wb'))

#datapath = [[['data/CMKP_data'],['train','valid','test']],[['data/Twitter_data'],['train','valid','test']],[['data/StackExchange_data'],['train','valid','test']]]
            # ,[['data/KP20K/kp20k_separated'],['train','valid']],[['data/KP20K/testsets/kp20k'],['test']],[['data/KP20K/testsets/inspec'],['test']],[['data/KP20K/testsets/krapivin'],['test']],
            # [['data/KP20K/testsets/nus'],['test']],[['data/KP20K/testsets/semeval'],['test']]]
datapath = [[['data/KP20K/kp20k_separated'],['train','valid']],[['data/KP20K/testsets/kp20k'],['test']],[['data/KP20K/testsets/inspec'],['test']],[['data/KP20K/testsets/krapivin'],['test']],[['data/KP20K/testsets/nus'],['test']],[['data/KP20K/testsets/semeval'],['test']]]
for path, datatypes in datapath:
    for datatype in datatypes:
            dataprocess(path[0], datatype)
