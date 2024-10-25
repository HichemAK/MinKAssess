# Statistical Knowledge Assessment for Large Language Models
A minimal implementation of KaRR knowledge assessment method from the following paper:

> [**Statistical Knowledge Assessment for Large Language Models**](https://arxiv.org/abs/2305.10519),            
> Qingxiu Dong, Jingjing Xu, Lingpeng Kong, Zhifang Sui, Lei Li   
> *arXiv preprint ([arxiv_version](https://arxiv.org/abs/2305.10519))*   

This is a fork of the [official implementation](https://github.com/dqxiu/KAssess) released by the authors.

## How to use?

First setup the conda environment using the following command

```bash
pip install minkarr
```

Here is a simple example of **how to quantify the knowledge of a fact by an LLM using KaRR**
```python
from karr import KaRR
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'gpt2'
device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(model_name, device_map = device)
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
# Output:
# KaRR = 3.338972442145268
# According to KaRR, does the model knows this fact? Answer: No
```

## Difference with original repo

- Easy-to-use
- Clean code
- Minimalistic implementation: I kept only the portion of the code needed to compute KaRR and removed the rest
- This implementation can compute KaRR on a single fact (the original implementation went through all facts)

## Citation
Cite the original authors using:
```
@misc{dong2023statistical,
      title={Statistical Knowledge Assessment for Large Language Models}, 
      author={Qingxiu Dong and Jingjing Xu and Lingpeng Kong and Zhifang Sui and Lei Li},
      year={2023},
      journal = {Proceedings of NeurIPS},
}
```



