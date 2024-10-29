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

paired_words = []

while len(words) > 1:
    pairs = list(permutations(words, 2))
    best_prob, best_words = 0, ('', '')
    for (i, j) in pairs:
        w = i + ' ' + j
        if best_prob == 0 or best_prob < sentence_prob(w):
            best_prob = sentence_prob(w)
            best_words = (i, j)
    # print(best_words, ' ', sentence_prob(best_words[0] + ' ' + best_words[1]))
    paired_words.append(best_words[0] + ' ' + best_words[1])
    words.remove(best_words[0])
    words.remove(best_words[1])
if len(words) > 0:    
    paired_words.append(words[0])
    words.clear()

print(paired_words)

ans = []
for p in permutations(paired_words):
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

# -- Wczoraj wieczorem spotkałem pewną wspaniałą kobietę, która z pasją opowiadała o modelach językowych.
# Wczoraj wieczorem pewną spotkałem która z wspaniałą pasją opowiadała kobietę o modelach językowych. -97.28199
# Wczoraj wieczorem pewną spotkałem kobietę o modelach językowych która z wspaniałą pasją opowiadała. -98.076416
# Pewną spotkałem wczoraj wieczorem która z wspaniałą pasją opowiadała kobietę o modelach językowych. -102.96242
# Kobietę o modelach językowych pewną spotkałem wczoraj wieczorem która z wspaniałą pasją opowiadała. -104.483734
# Pewną spotkałem wczoraj wieczorem kobietę o modelach językowych która z wspaniałą pasją opowiadała. -105.41408
# Wczoraj wieczorem pewną spotkałem kobietę o która z wspaniałą pasją opowiadała modelach językowych. -105.46166
# Wczoraj wieczorem pewną spotkałem która z wspaniałą pasją kobietę o modelach językowych opowiadała. -105.54963
# Wczoraj wieczorem pewną spotkałem opowiadała która z wspaniałą pasją kobietę o modelach językowych. -106.45133
# Wczoraj wieczorem która z wspaniałą pasją opowiadała pewną spotkałem kobietę o modelach językowych. -106.5515
# Która z wspaniałą pasją opowiadała pewną spotkałem wczoraj wieczorem kobietę o modelach językowych. -106.5721