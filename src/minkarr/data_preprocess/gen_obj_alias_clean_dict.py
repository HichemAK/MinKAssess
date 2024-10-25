'''
Preprocess for OOV words in the object aliases.
'''

import json
from transformers import GPT2TokenizerFast, AutoTokenizer, TransfoXLTokenizer, T5Tokenizer, OpenAIGPTTokenizer, XLNetTokenizer, AutoTokenizer
from transformers import GPT2Tokenizer, LlamaTokenizer

from transformers import AutoTokenizer

from minkarr.globs import PROJECT_PATH

def judge_obj_in_vocab(tokenizer, obj_label, obj_ids, verbose=0):

    if isinstance(tokenizer, GPT2TokenizerFast):
        reconstructed_word = "".join(
            tokenizer.convert_ids_to_tokens(obj_ids)).replace('Ġ', ' ').strip()
    elif isinstance(tokenizer, TransfoXLTokenizer):
        reconstructed_word = " ".join(
            tokenizer.convert_ids_to_tokens(obj_ids)).replace(' , ', ', ').strip()
    elif type(tokenizer).__name__ == 'GLMGPT2Tokenizer':
        reconstructed_word = tokenizer.decode(obj_ids, skip_special_tokens=True).strip()
    elif isinstance(tokenizer, LlamaTokenizer):
        reconstructed_word = tokenizer.decode(obj_ids, skip_special_tokens=True).strip()
    elif isinstance(tokenizer, T5Tokenizer):
        reconstructed_word = tokenizer.decode(obj_ids, skip_special_tokens=True).strip()
    elif isinstance(tokenizer, OpenAIGPTTokenizer):
        reconstructed_word = tokenizer.decode(obj_ids, clean_up_tokenization_spaces=True)
    elif isinstance(tokenizer, XLNetTokenizer) or isinstance(tokenizer, GPT2Tokenizer):
        reconstructed_word = tokenizer.decode(obj_ids, skip_special_tokens=True)
    else:
        reconstructed_word = "".join(tokenizer.convert_ids_to_tokens(obj_ids)).replace('Ġ', ' ').strip()
    if isinstance(tokenizer, OpenAIGPTTokenizer) or isinstance(tokenizer, GPT2Tokenizer): 
        if (not reconstructed_word) or (reconstructed_word.lower().replace(' ','') != obj_label.lower().replace(' ','')):
            if verbose:
                print("\tEXCLUDED object label {} not in model vocabulary\n".format(
                    obj_ids
                ))
            return False
        return True
    else: 
        if (not reconstructed_word) or (reconstructed_word != obj_label):
            if verbose:
                print("\tEXCLUDED object label {} not in model vocabulary\n".format(
                    obj_ids
                ))
            return False
        return True

if __name__ == '__main__':
    device = "cuda"
    model_names = ['EleutherAI/gpt-neo-125M']
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
        with open(f"{PROJECT_PATH}/data/cleaned_T_REx/allobj2alias.json",'r') as load_f:
            obj2alias = json.load(load_f)
            
        save_dict = {}
        valid_obj_num = 0
        for obj_id in obj2alias.keys():
            origial_aliases = obj2alias[obj_id]
            save_dict[obj_id] = []
            for alias in origial_aliases:
                if alias == None:
                    continue
                input_ids = tokenizer(alias, return_tensors='pt').to(device).input_ids[0]
                if judge_obj_in_vocab(tokenizer, alias, input_ids):
                    save_dict[obj_id].append(alias)
                    valid_obj_num += 1
                else: 
                    print(f"alias: {alias}, judge: {judge_obj_in_vocab(tokenizer, alias, input_ids)}")
        model_name_replaced = model_name.replace('/', '_')
        with open(f"{PROJECT_PATH}/data/cleaned_T_REx/obj2alias_for_{model_name_replaced}_vocab.json", 'w') as write_f:
            json.dump(save_dict, write_f, indent=4, ensure_ascii=False)
        print(valid_obj_num / len(save_dict.keys()))