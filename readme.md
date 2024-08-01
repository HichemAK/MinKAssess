# Statistical Knowledge Assessment for Large Language Models
A minimal implementation of KaaR knowledge assessment method from the following paper:

> [**Statistical Knowledge Assessment for Large Language Models**](https://arxiv.org/abs/2305.10519),            
> Qingxiu Dong, Jingjing Xu, Lingpeng Kong, Zhifang Sui, Lei Li   
> *arXiv preprint ([arxiv_version](https://arxiv.org/abs/2305.10519))*   

This is a fork of the [official implementation](https://github.com/dqxiu/KAssess) released by the authors.

## How to use?

First setup the conda environment using the following command

```bash
conda env create -f environment.yml
conda init bash
conda activate KAssess
```

Here is a simple example of how to quantify the knowledge of a fact by an LLM using KaaR
```python
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
# Output:
# KaaR = 3.338972442145268
# According to KaaR, does the model knows this fact? Answer: No
```

## Difference with original repo

- Easy-to-use
- Clean code
- Minimalistic implementation: I kept only the portion of the code needed to compute KaaR and removed the rest
- This implementation can compute KaaR on a single fact (the original implementation went through all facts)

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



