import json
from statistics import geometric_mean

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BertForMaskedLM, BertTokenizerFast, AutoTokenizer, TransfoXLTokenizer, TransfoXLLMHeadModel,T5Tokenizer, T5ForConditionalGeneration,OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, XLNetTokenizer, XLNetLMHeadModel, GPTNeoForCausalLM, AutoTokenizer, GPTJForCausalLM,AutoModelForCausalLM
from transformers import AutoTokenizer, BloomForCausalLM, OPTForCausalLM, AutoModelForSeq2SeqLM
from code.data_preprocess.wikidata_get import *
import random
import jsonlines
import os
import argparse
import nltk
import math

from globs import PROJECT_PATH
from utils import load_json
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')
from nltk.corpus import stopwords

# set_start_method('spawn')
ctx = torch.multiprocessing.get_context("spawn")
random.seed(1)


def data_filter(probing_facts, rel2alias, sub2alias, obj2alias):
    filtered_facts = dict()
    print("filtering data")
    for fact_id in probing_facts.keys():
        fact = probing_facts[fact_id]
        sub_id = fact[0]
        rel_id = fact[1]
        obj_id = fact[2]

        sub_gold_info = get_entity_defaut_alias(sub_id, sub2alias)
        obj_gold_info = get_entity_defaut_alias(obj_id, obj2alias)
        if sub_gold_info == None or obj_gold_info == None:
            print("no defaut alias for sub or obj")
            continue
        sub_gold = sub_gold_info
        obj_gold = obj_gold_info
        
        if rel_id not in rel2alias.keys():
            print("no single text rel")
            continue
        
        rel_gold_list = rel2alias[rel_id]

        filtered_facts[fact_id] = [sub_id, rel_id, obj_id, sub_gold, rel_gold_list, obj_gold]
        if sub_id not in sub2alias.keys():
            sub2alias[sub_id] = [sub_gold]
        elif sub_gold not in sub2alias[sub_id]:
            sub2alias[sub_id].append(sub_gold)
        if obj_id not in obj2alias.keys():
            obj2alias[obj_id] = [obj_gold]
        elif obj_gold not in obj2alias[obj_id]:
            obj2alias[obj_id].append(obj_gold)
        
    return filtered_facts, sub2alias, obj2alias

    
def gmean(input_x, dim):
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))

def get_entity_defaut_alias(entity_id, whole_dict):
    if entity_id in whole_dict.keys() and len(whole_dict[entity_id])>0:
        return whole_dict[entity_id][0]
    else:
        return None
    

def condition_prob(predict_sent, given_sent, tokenizer, model, device, mode="given beta"):
    if mode == "given beta":
        # predict_sent = o
        prompt = predict_sent # s,r
        tgt_len = len(tokenizer.encode(' ' + predict_sent.replace(given_sent, '').strip()))
        encodings = tokenizer(prompt,return_tensors='pt')
        input_ids = encodings.input_ids.to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-tgt_len] = -100
        outputs = model(input_ids, labels=target_ids)
        ppl = torch.exp(outputs.loss)
        return 1 / ppl.item()
    
    if mode == "given alpha":
        # predict_sent = r,o
        prompt = predict_sent # s
        tgt_len = len(tokenizer.encode(' ' + predict_sent.replace(given_sent, '').strip()))
        encodings = tokenizer(prompt,return_tensors='pt')
        input_ids = encodings.input_ids.to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-tgt_len] = -100
        outputs = model(input_ids, labels=target_ids)
        ppl = torch.exp(outputs.loss)
        return 1 / ppl.item()
    

def sentence_prob(sentence,  tokenizer, model, device):
    encodings = tokenizer(sentence, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)
    target_ids = input_ids.clone()
    outputs = model(input_ids, labels=target_ids)
    ppl = torch.exp(outputs.loss)
    prob = 1 / ppl.item()
    return prob


def rr_bs_sub_replace(tokenizer, model, device, sub_id, rel_id, obj_id, sub2alias, rel2alias, obj2alias, rel2sub2rate):
    '''
    Given rel sub replacement
    score = 
    '''


    betas = []
    beta_temps = []
    gammas = []
    alphas = sub2alias[sub_id]
    for r_alias in rel2alias[rel_id]:
        r_alias = r_alias.strip('.').strip()
        if len(r_alias) == 0:
            continue
        for alpha in alphas:
            betas.append(r_alias.replace('[X]', alpha))
        beta_temps.append(r_alias.strip('.').strip())
    
    
    for o_alias in obj2alias[obj_id]:
        for beta in betas:
            gammas.append(beta.replace('[Y]', o_alias))
            
    betas = [beta.replace('[Y]', '').strip() for beta in betas]
    
    
    with torch.no_grad():

        # sum_beta [P_M(beta) * \delta(s,r) * sum_gamma P_M(gamma|beta) * \delta(o)]
        # betas represents aliases of s,r
        p_numerator_info = dict()
        p_denominator_info = dict()
        
        p_numerator = 0
        for beta in betas:
            # P_M(beta)
            p_beta = sentence_prob(beta, tokenizer, model, device)
            p_gamma_sum = 0
            gamma_dict = dict()
            for gamma in gammas:
                # P_M(gamma|beta)
                if beta not in gamma:
                    continue
                p_gamma = condition_prob(gamma, beta, tokenizer, model, device, mode="given beta")
                gamma_dict[gamma] = round(p_gamma, 6)
                p_gamma_sum += p_gamma
            if math.isnan(p_beta) or math.isnan(p_gamma_sum):
                continue
            p_numerator += p_beta * p_gamma_sum

        p_numerator_info[beta] = { 'p_beta': round(p_beta, 6), 'p_gamma_sum': round(p_gamma_sum, 6), 'gamma_dict': gamma_dict}
        
        p_denominator = 0
        subids_list = [other_sub_id for other_sub_id in rel2sub2rate[rel_id].keys()] 
        # Q_weights = torch.Tensor(probs_list)
        sample_k = 4
        # Q_sampled_sub_idxes = torch.multinomial(Q_weights, min(len(probs_list),sample_k))
        Q_sampled_sub_idxes = random.sample(range(len(subids_list)), min(sample_k, len(subids_list)))
        Q_sampled_sub_ids = [subids_list[idx] for idx in Q_sampled_sub_idxes]
        for beta_temp in beta_temps:
            for other_sub_id in Q_sampled_sub_ids:
                for alpha in sub2alias[other_sub_id]:
                    if alpha == None or beta_temp == None:
                        continue
                    p_beta = sentence_prob(beta_temp.replace('[X]', alpha).replace('[Y]','').strip(), tokenizer, model, device)
                    # Q = rel2sub2rate[rel_id][other_sub_id] * (1/len(sub2alias[other_sub_id]))
                    # print(f"alpha: {alpha}")
                    # P_m = sentence_prob(alpha, tokenizer, model, device)
                    p_gamma_sum = 0
                    gamma_dict = dict()
                    for o_alias in obj2alias[obj_id]:
                        gamma = beta_temp.replace('[X]', alpha).replace('[Y]', o_alias)
                        p_gamma = condition_prob(gamma, beta_temp.replace('[X]', alpha).replace('[Y]','').strip(), tokenizer, model, device, mode="given beta")
                        p_gamma_sum += p_gamma
                        gamma_dict[gamma] = round(p_gamma, 6)
                    p_denominator += p_beta * p_gamma_sum
        p_denominator = p_denominator / min(sample_k, len(subids_list))
        p_denominator_info[beta] = {'p_beta': round(p_beta, 6), 'p_gamma_sum': round(p_gamma_sum, 6), 'gamma_dict': gamma_dict}
        
        if p_denominator == 0:
            return None
        rr_result = {"score": p_numerator / p_denominator, 'p_numerator_score': p_numerator, "p_numerator_info": p_numerator_info, 'p_denominator_score': p_denominator, "p_denominator_info": p_denominator_info}
        # print(rr_result)
    return rr_result

def rr_bs_rel_replace(tokenizer, model, device, sub_id, rel_id, obj_id, sub2alias, rel2alias, obj2alias, obj2rel2rate):
    '''
    Given sub, rel replacement
    '''
    betas = []
    gammas = []
    alphas = sub2alias[sub_id]
    for r_alias in rel2alias[rel_id]:
        r_alias = r_alias.strip('.').strip()
        if len(r_alias) == 0:
            continue
        for alpha in alphas:
            betas.append(r_alias.replace('[X]', alpha))
    
    
    for o_alias in obj2alias[obj_id]:
        for beta in betas:
            gammas.append(beta.replace('[Y]', o_alias))
            
    betas = [beta.replace('[Y]', '').strip() for beta in betas]
    
    with torch.no_grad():

        # sum_beta [P_M(beta) * \delta(s,r) * sum_gamma P_M(gamma|beta) * \delta(o)]
        # betas represents aliases of s,r
        p_numerator_info = dict()
        p_denominator_info = dict()
        
        p_numerator = 0
        for beta in betas:
            # P_M(beta)
            p_beta = sentence_prob(beta, tokenizer, model, device)
            p_gamma_sum = 0
            gamma_dict = dict()
            for gamma in gammas:
                # P_M(gamma|beta)
                if beta not in gamma:
                    continue
                p_gamma = condition_prob(gamma, beta, tokenizer, model, device, mode="given beta")
                gamma_dict[gamma] = round(p_gamma, 6)
                p_gamma_sum += p_gamma
            if math.isnan(p_beta) or math.isnan(p_gamma_sum):
                continue
            p_numerator += p_beta * p_gamma_sum

        p_numerator_info[beta] = { 'p_beta': round(p_beta, 6), 'p_gamma_sum': round(p_gamma_sum, 6), 'gamma_dict': gamma_dict}
        
        p_denominator = 0
        relids_list = list(obj2rel2rate[obj_id].keys())
        sample_k = 4
        Q_sampled_rel_idxes = random.sample(range(len(relids_list)), min(sample_k, len(relids_list)))
        Q_sampled_rel_ids = [relids_list[idx] for idx in Q_sampled_rel_idxes]
        for alpha in alphas:
            # P_M(alpha)
            p_alpha = sentence_prob(alpha, tokenizer, model, device)
            p_gamma_sum = 0
            gamma_dict = dict()
            for other_rel_id in Q_sampled_rel_ids:
                # random sample from rel2alias[other_rel_id]
                beta = random.sample(rel2alias[other_rel_id], 1)[0]
            # for beta_text in all_r_gammas[alpha]:
                # P_M(gamma|beta) sum P(Obama was born in Hawaii | Obama) all possible strings
                for o_alias in obj2alias[obj_id]:
                    if beta==None or alpha==None or o_alias==None:
                        continue
                    gamma = beta.replace('[X]', alpha).replace('[Y]', o_alias) 
                    p_gamma = condition_prob(gamma, alpha, tokenizer, model, device, mode="given alpha")
                    p_gamma_sum += p_gamma
                    gamma_dict[gamma] = round(p_gamma, 6)
            if math.isnan(p_alpha) or math.isnan(p_gamma_sum):
                continue
            p_denominator += p_alpha * p_gamma_sum 
        p_denominator = p_denominator / (min(sample_k, len(relids_list)))
        p_denominator_info[beta] = {'p_alpha': round(p_alpha, 6), 'p_gamma_sum': round(p_gamma_sum, 6), 'gamma_dict': gamma_dict}
        
        if p_denominator == 0:
            return None
        rr_result = {"score": p_numerator / p_denominator, 'p_numerator_score': p_numerator, "p_numerator_info": p_numerator_info, 'p_denominator_score': p_denominator, "p_denominator_info": p_denominator_info}
        # print(rr_result)
    return rr_result


def build_fact_res(fact, tokenizer, model, device, sub2alias, rel2alias, obj2alias, obj2rel2rate, rel2sub2rate):
    sub_id, rel_id, obj_id = fact

    
    rr_bs_sub_result = rr_bs_sub_replace(tokenizer, model, device, sub_id, rel_id, obj_id, sub2alias, rel2alias, obj2alias, rel2sub2rate)
    rr_bs_rel_result = rr_bs_rel_replace(tokenizer, model, device, sub_id, rel_id, obj_id, sub2alias, rel2alias, obj2alias, obj2rel2rate)
    
    if rr_bs_sub_result is None or rr_bs_rel_result is None:
        raise Exception
        
    fact_res = {'rr_bs_sub_result': rr_bs_sub_result, 'rr_bs_rel_result': rr_bs_rel_result}
    return fact_res

def load_model(model_name, device="cuda"):
    if 'gpt2' in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    elif 'bloom' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BloomForCausalLM.from_pretrained(model_name).to(device)
    elif 'opt' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
    elif 'llama' in model_name and 'alpaca' not in model_name:
        from transformers import LlamaForCausalLM
        if '65B' or '30B' or '7B' in model_name:
            tokenizer = AutoTokenizer.from_pretrained("YOUR_MODEL_PATH/pretrained_models/LLAMA-hf/{}".format(model_name.split('-')[-1]), use_fast=False)
            model = LlamaForCausalLM.from_pretrained("YOUR_MODEL_PATH/pretrained_models/LLAMA-hf/{}".format(model_name.split('-')[-1]), torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained("YOUR_MODEL_PATH/pretrained_models/LLAMA-hf/{}".format(model_name.split('-')[-1]), use_fast=False)
            model = LlamaForCausalLM.from_pretrained("YOUR_MODEL_PATH/pretrained_models/LLAMA-hf/{}".format(model_name.split('-')[-1]), torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    elif 'alpaca' in model_name:
        from transformers import LlamaForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    elif 'vicuna' in model_name:
        from transformers import LlamaForCausalLM
        if '7b' or '13b' in model_name:
            tokenizer = AutoTokenizer.from_pretrained("{}pretrained_models/{}".format(PROJECT_PATH, model_name), use_fast=False)
            model = LlamaForCausalLM.from_pretrained("{}pretrained_models/{}".format(PROJECT_PATH, model_name), torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
    elif 'glm' in model_name:
        tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-10b", trust_remote_code=True)
        model =  AutoModelForSeq2SeqLM.from_pretrained("THUDM/glm-10b", trust_remote_code=True).to(device)
    elif 'moss' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).half().cuda()
    elif 'dolly' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    elif 'bert' in model_name:
        model = BertForMaskedLM.from_pretrained(model_name).to(device)
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
    elif 't5' in model_name:
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    elif 'openai-gpt' in model_name:
        model = OpenAIGPTLMHeadModel.from_pretrained(model_name).to(device)
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name)
    elif 'xlnet' in model_name:
        model = XLNetLMHeadModel.from_pretrained(model_name).to(device)
        tokenizer = XLNetTokenizer.from_pretrained(model_name)
    elif 'gpt-j' in model_name:
        model = GPTJForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif 'neo' in model_name:
        model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
        model = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103").to(device)
    model.eval()
    return model, tokenizer

def load_data(model_name_replaced):
    print("⏰ Loading data....")
    rootdir = f"{PROJECT_PATH}/data/my_TREx_main_new"
        
        
    all_trex = []
    list_path = os.listdir(rootdir)
    for i in range(0, len(list_path)):
        # Construction path
        path = os.path.join(rootdir, list_path[i])
        with jsonlines.open(path) as reader:
            for obj in reader:
                all_trex.append(obj)
    
    # with open(f"{PROJECT_PATH}/data/symbol2text.json",
    #             'r') as load_f:
    #     rel_dict = json.load(load_f)
    # with open(f"{PROJECT_PATH}/data/cleaned_T_REx/rel2sub_ids.json",
    #             'r') as load_f:
    #     rel2sub_ids = json.load(load_f)
    with open(f"{PROJECT_PATH}/data/cleaned_T_REx/rel2sub2rate.json",
                'r') as load_f:
        rel2sub2rate = json.load(load_f)
    # single_tok_objdict = load_json(f"{PROJECT_PATH}/data/cleaned_T_REx/single_tok_objdict.json")
    if 'gpt2' in model_name_replaced:
        vocab_path = f"{PROJECT_PATH}/data/cleaned_T_REx/obj2alias_for_gpt2_vocab.json"
    elif 't5' in model_name_replaced:
        vocab_path = f"{PROJECT_PATH}/data/cleaned_T_REx/obj2alias_for_t5-large_vocab.json"
    elif 'bloom' in model_name_replaced:
        vocab_path = f"{PROJECT_PATH}/data/cleaned_T_REx/obj2alias_for_bigscience_bloom-560m_vocab.json"
    elif 'opt' in model_name_replaced:
        vocab_path = f"{PROJECT_PATH}/data/cleaned_T_REx/obj2alias_for_facebook_opt-125m_vocab.json"
    elif 'llama' in model_name_replaced or 'alpaca' in model_name_replaced or 'vicuna' in model_name_replaced:
        vocab_path = f"{PROJECT_PATH}/data/cleaned_T_REx/obj2alias_for_decapoda-research_llama-7b-hf_vocab.json"
    elif 'glm' in model_name_replaced or "moss" in model_name_replaced:
        vocab_path = f"{PROJECT_PATH}/data/cleaned_T_REx/obj2alias_for_glm_vocab.json"
    else: 
        vocab_path = f"{PROJECT_PATH}/data/cleaned_T_REx/obj2alias_for_{model_name_replaced}_vocab.json"

    obj2alias = load_json(vocab_path)
    if 'openai-opt' in model_name_replaced:
        for obj_id in obj2alias.keys():
            cur_obj_aliases = obj2alias[obj_id]
            for obj_alias_id in range(len(cur_obj_aliases)):
                cur_obj_aliases[obj_alias_id] = cur_obj_aliases[obj_alias_id].lower()
    sub2alias = load_json(f"{PROJECT_PATH}/data/cleaned_T_REx/allsub2alias.json")
    obj2rel2rate = load_json(f"{PROJECT_PATH}/data/cleaned_T_REx/obj2rel2rate.json")
    # rel alias filter
    rel2alias = load_json(f"{PROJECT_PATH}/data/relation2template.json")
    print("😄 All data loaded.\n")
    return all_trex, sub2alias, rel2alias, obj2alias, obj2rel2rate, rel2sub2rate

def compute_kaar(model, tokenizer, fact, device) -> tuple[float, bool]:
    stopwords = stopwords.words('english')
    stopwords.extend(['I', 'J', 'K', 'without'])
    model_name = model.config.name
    model_name_replaced = model_name.replace('/', '_')
    
    all_trex, sub2alias, rel2alias, obj2alias, obj2rel2rate, rel2sub2rate = load_data(model_name_replaced)
    
    sample_facts_num = len(all_trex) 
    sample_trex_items_ids = random.sample(range(len(all_trex)), sample_facts_num)
    sample_trex_items = dict()
    for sample_id in sample_trex_items_ids:
        if len(sample_trex_items.keys()) > sample_facts_num:
            break
        
        fact_id = all_trex[sample_id]['fact_id']
        item = all_trex[sample_id]
        if 'sub_label' in item.keys():
            # for false facts
            false_trans = {'Atlanta': 'Q23556', 'English': 'Q1860', 'Chicago': 'Q1297', 'France': 'Q142', 'London': 'Q84', 'French': 'Q150', 'medicine': 'Q11190', 'United': 'Q30', 'NBC': 'Q13974', 'RCA': 'Q50074604', 'ABC': 'Q169889', 'England': 'Q21', 'science': 'Q336'}
            if '#' in false_trans[item['obj_label']] or '#' in item['subj_label']:
                print("the fact is not in the original T_REx dataset")
                continue
            sample_trex_items[str(fact_id)] = (item['subj_label'], item['relation'], false_trans[item['obj_label']])
        else:
            if '#' in item['obj_label'] or '#' in item['subj_label']:
                print("the fact is not in the original T_REx dataset")
                continue
            sample_trex_items[str(fact_id)] = (item['subj_label'], item['relation'], item['obj_label'])
            
        

    _, sub2alias, obj2alias = data_filter(sample_trex_items, rel2alias, sub2alias, obj2alias)
    for rel in rel2alias:
        if len(rel2alias[rel]) > 0:
            rel2alias[rel] = random.sample(rel2alias[rel], min(4, len(rel2alias[rel])))
    for sub in sub2alias:
        if len(sub2alias[sub]) > 0:
            sub2alias[sub] = random.sample(sub2alias[sub], min(4, len(sub2alias[sub])))
    for obj in obj2alias:
        if len(obj2alias[obj]) > 0:
            obj2alias[obj] = random.sample(obj2alias[obj], min(4, len(obj2alias[obj])))
            
    fact_res = build_fact_res(fact, tokenizer, model, device, sub2alias, rel2alias, obj2alias, obj2rel2rate, rel2sub2rate)
    return get_kaar(fact_res)

def get_kaar(fact_res : dict) -> tuple[float, bool]:
    thresh = 22
    # I need load_dict
    rr_sub_result = fact_res["rr_bs_sub_result"]
    rr_rel_result = fact_res["rr_bs_rel_result"]
    if float(rr_sub_result['score'])==0.0:
        rr_sub_result['score'] = 0.000001
    if float(rr_rel_result['score'])==0.0:
        rr_rel_result['score'] = 0.000001
    cur_birr = geometric_mean([float(rr_sub_result['score']),float(rr_rel_result['score'])])
    does_know = cur_birr > thresh
    return cur_birr, does_know

if __name__ == '__main__':
    model_name = 'gpt2-xl'
    device = 'cuda'
    model, tokenizer = load_model(model_name, device)

    # fact = (France, capital, Paris)
    fact = ('Q142', 'P36', 'Q90')

    kaar, does_know = compute_kaar(model, tokenizer, fact, device)
    print('Fact %s' % fact)
    print('KaaR = %s, does_know = %s' % (kaar, does_know))