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
last_prompt = 'Jak się czujesz?'

answer_beginnings = ['Uważam, że', 'W większości przypadków', 'Moim zdaniem', 'Odpowiedzią, na to pytanie jest']

while True:
    prompt = input().strip()
    prompt_input = prompt
    if prompt.lower() in ['exit', 'quit', 'stop']:
        print("Czatbot: Do zobaczenia!")
        break
    if not prompt:
        prompt = last_prompt
    else:
        prompt = last_prompt + ' ' + prompt
        #prompt = prompt[-50:]
    print("prompt = ", prompt)
    g = generator(prompt, pad_token_id=generator.tokenizer.eos_token_id, max_length=100, truncation=True)[0]['generated_text']
    last_dot_index = g.rfind('.')
    if last_dot_index != -1:
        g = g[:last_dot_index+1]

    # print(g, sentence_prob(g))
    # print(50 * '=')
    # print()

    best_prompt, best_prob = g, sentence_prob(g)

    for ab in answer_beginnings:
        new_prompt = prompt + ' ' + ab
        g = generator(new_prompt, pad_token_id=generator.tokenizer.eos_token_id, max_length=100, truncation=True)[0]['generated_text']
        last_dot_index = g.rfind('.')
        if last_dot_index != -1:
            g = g[:last_dot_index+1]

        if best_prob < sentence_prob(g):
            best_prompt = g
            best_prob = sentence_prob(g)

        # print(g, sentence_prob(g))
        # print(50 * '=')
        # print()
    print(best_prompt, best_prob)
    print(50 * '=')
    print()
    last_prompt = prompt_input