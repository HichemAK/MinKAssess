from minkarr import KaRR
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'gpt2'
device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)

karr = KaRR(model, tokenizer, device)

# Testing the fact: (France, capital, Paris)
# You can find other facts by looking into Wikidata
fact = ('Q142', 'P36', 'Q90')

karr, does_know = karr.compute(fact)
print('Fact %s' % str(fact))
print('KaRR = %s' % karr)
ans = 'Yes' if does_know else 'No'
print('According to KaRR, does the model knows this fact? Answer: %s' % ans)
