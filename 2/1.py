import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from torch.nn import functional as F
import transformers
import re

# generator = pipeline('text-generation', model='flax-community/papuGaPT2', device=0)
# model_name = 'flax-community/papuGaPT2'
generator = pipeline('text-generation', model='eryk-mazus/polka-1.1b-chat', device=0)
model_name = 'eryk-mazus/polka-1.1b-chat'
transformers.logging.set_verbosity_error()
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

intro = 'Obliczamy działania arytmetyczne. Jeden dodać jeden równa się dwa. Trzy dodać cztery równa się siedem. Pięć dodać dwa równa się siedem. '

# expressions = ['Jeden dodać jeden', 'Jeden dodać dwa', 'Jeden dodać trzy', 'Jeden dodać cztery', 'Jeden dodać pięć', 'Jeden dodać sześć', 'Jeden dodać sidem', 'Jeden dodać osiem', 'Jeden dodać dziewięć', 'Dwa dodać dwa', 'Dwa dodać trzy', 'Dwa dodać cztery', 'Dwa dodać pięć', 'Dwa dodać sześć', 'Dwa dodać siedem', 'Dwa dodać osiem', 'Dwa dodać dziewięć', 'Trzy dodać trzy', 'Trzy dodać cztery', 'Trzy dodać pięć']
# results = ['dwa', 'trzy', 'cztery', 'pięć', 'sześć', 'siedem', 'osiem', 'dziewięć', 'dziesięć', 'cztery', 'pięć', 'sześć', 'siedem', 'osiem', 'dziewięć', 'dziesięć', 'jedenaście','sześć', 'siedem', 'osiem']

expressions = ['1 + 1', '1 + 2', '1 + 3', '1 + 4', '1 + 5', '1 + 6', '1 + 7', '1 + 8', '1 + 9', '2 + 2', '2 + 3', '2 + 4', '2 + 5', '2 + 6', '2 + 7', '2 + 8', '2 + 9', '3 + 3', '3 + 4', '3 + 5']
results = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '4', '5', '6', '7', '8', '9', '10', '11', '6', '7', '8']

def get_first_word(text):
    match = re.match(r'\w+', text)
    return match.group(0) if match else None

def get_first_number(text):
    result = ''
    for i in text:
        if i not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            break
        result = result + i
    return result

# prompt = input().strip()
correct, idx = 0, 0
for e in expressions:
    # full_prompt = intro + e + ' równa się '
    full_prompt = e + ' = '
    g = generator(full_prompt, pad_token_id=generator.tokenizer.eos_token_id, max_length=10, truncation=True)[0]['generated_text']
    # print(get_first_number(str(g[len(full_prompt)+1:])) + ' ' + str(results[idx]))
    print(get_first_number(g[len(full_prompt):]), end='')
    print(' ', end='')
    print(results[idx])
    # print(g[len(intro):])
    if(results[idx] == get_first_number(g[len(full_prompt):])):
        correct += 1
        # print(expressions[idx] + ' równa się ' + results[idx])
    idx += 1
print(str(correct) + '/' + str(len(expressions)))

# dodawnaie zapisane cyframi:
# średnio 16/20 ~ 80% poprawnych obiczeń


# dodawanie zapisane słownie:
# dodajemy dwie cyfry tak by wynik dodawania mieścił się w przediale [2, 11], czyli przy losowym generowaniu odpowiedzi mielibyśmy 1/10 prawdopodobieństwo na poprawy wynik 
# dla przykładowych wyrazen oczekiwana liczba poprawych odpowiedzi wynosi EX = 20*1/10 = 2
# wyniki modelu poprawnie odpowiadały średnio na 6 działań
