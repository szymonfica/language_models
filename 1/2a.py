import re
from itertools import permutations
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from torch.nn import functional as F

generator = pipeline('text-generation', model='flax-community/papuGaPT2', device=0)
model_name = 'flax-community/papuGaPT2'
device = 'cuda'
device = 'cpu'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def log_probs_from_logits(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logp_label
    
            
def sentence_prob(sentence_txt):
    input_ids = tokenizer(sentence_txt, return_tensors='pt')['input_ids'].to(device)
    with torch.no_grad():
        output = model(input_ids=input_ids)
        log_probs = log_probs_from_logits(output.logits[:, :-1, :], input_ids[:, 1:])
        seq_log_probs = torch.sum(log_probs)
    return seq_log_probs.cpu().numpy()

print('Model loaded')

def split_polish_sentence(sentence):
    return re.sub(r'[^\w\s]', '', sentence).lower().split()

sentence = input()
words = split_polish_sentence(sentence)
ans = []
for p in permutations(words):
    s = ' '.join(p) + '.'
    s = s.capitalize()
    # print(f'{s} {sentence_prob(s)}')
    ans.append((sentence_prob(s), s))
ans.sort(reverse=True)
limit = 10
for a in ans:
    print(a[1], a[0])
    limit -= 1
    if limit == 0:
        break
