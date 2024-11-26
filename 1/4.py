import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from torch.nn import functional as F
import transformers
transformers.logging.set_verbosity_error()


generator = pipeline('text-generation', model='flax-community/papuGaPT2', device=0)
model_name = 'flax-community/papuGaPT2'
# generator = pipeline('text-generation', model='eryk-mazus/polka-1.1b-chat', device=0)
# model_name = 'eryk-mazus/polka-1.1b-chat'
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

with open('questions.txt', 'r') as q_file, open('found_answers.txt', 'w') as a_file:
    idx = 5
    for q in q_file:
        if(idx <= 0):
            break
        idx -= 1
        question = q.strip()
        answer = ''

        if question.startswith('Czy'):
            yes_prob = normalized_sentence_prob(question + ' tak')
            no_prob = normalized_sentence_prob(question + ' nie')
            if yes_prob < no_prob:
                # print('tak')
                answer = 'tak'
            else:
                # print('no')
                answer = 'nie'
        elif question.startswith('Ile'):
            prompt = 'Ile jest pór roku? Cztery. Ile razy jest planet w układzie słonecznym? W układzie słonecznym jest osiem planet. Ile lat trzeba mieć, by być pełnoletnim? Osiemnaście. ' + question 
            g = generator(prompt, pad_token_id=generator.tokenizer.eos_token_id, max_new_length=20, truncation=True)[0]['generated_text']
            g = g.split('.')[0]
            # print(g)
            answer = g
        # elif question.startswith('Jak'):
        #     prompt = 'Odpowiedź na pytanie: ' + question 
        #     g = generator(prompt, pad_token_id=generator.tokenizer.eos_token_id, max_length=100, truncation=True)[0]['generated_text']
        #     g = g[len(prompt)+1:]
        #     g = g.split('.')[0]
        #     # print(g)
        #     answer = g
        # elif question.startswith('Kto'):
        #     prompt = 'Odpowiedź na pytanie: ' + question 
        #     g = generator(prompt, pad_token_id=generator.tokenizer.eos_token_id, max_length=100, truncation=True)[0]['generated_text']
        #     g = g[len(prompt)+1:]
        #     g = g.split('.')[0]
        #     # print(g)
        #     answer = g
        else:
            prompt = 'Odpowiedź na pytanie: ' + question 
            g = generator(prompt, pad_token_id=generator.tokenizer.eos_token_id, max_length=70, truncation=True)[0]['generated_text']
            g = g[len(prompt)+1:]
            g = g.split('.')[0]
            # print(g)
            answer = g
        # print(idx, end=' ')
        # print(answer)
        a_file.write(str(idx) + ' ' + answer + '\n')

print('All questions answered.')