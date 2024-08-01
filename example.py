from kaar import KaaR

model_name = 'gpt2'
device = 'cuda'
kaar = KaaR(model_name, device)

# Testing the fact: (France, capital, Paris)
# You can find other facts by looking into Wikidata
fact = ('Q142', 'P36', 'Q90')

kaar, does_know = kaar.compute(fact)
print('Fact %s' % str(fact))
print('KaaR = %s' % kaar)
ans = 'Yes' if does_know else 'No'
print('According to KaaR, does the model knows this fact? Answer: %s' % ans)