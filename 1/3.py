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

def normalized_sentence_prob(txt):
    length = len(tokenizer(txt, return_tensors='pt')['input_ids'][0])
    return sentence_prob(txt) / length

print('Model loaded')
last_prompt = 'Super'

file = open('reviews_for_task3.txt', 'r')

count = [0, 0]
for line in file:
    if line.startswith('GOOD'):
        prompt = line[5:]
    elif line.startswith('BAD'):
        prompt = line[4:]
    else:
        continue
    good_prompt = prompt + ' Polecam'
    bad_prompt = prompt + ' Nie polecam'
    n_good_prob = normalized_sentence_prob(good_prompt)
    n_bad_prob = normalized_sentence_prob(bad_prompt)
    if n_good_prob > n_bad_prob:
        if line.startswith('GOOD'):
            count[0] += 1
    else:
        if line.startswith('BAD'):
            count[1] += 1
print(count[0], '/ 200 - positive reviews guessed correctly')
print(count[1], '/ 200 - negative reviews guessed correctly')

file.close()