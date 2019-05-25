from model_sampler import *

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import trange

from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer

def wrap_message_list(m_list, insert_intro=True, wrap_type='name', check_end_punct=True):
    '''
    Parameters:
    ----------
    m_list : list
        list of messages in chatbot log 
    insert_intro : bool, optional
        whether should it insert the intro about the conversation
    wrap_type : string, optional
        type of conditioning to use ('name', 'name-in-par', 'dash', 'number') 
    check_end_punct : bool, optional
        whether should it check the last symbol of message to have the period etc
    '''
    output = ""
    types = {'name': ('Alice: ', 'Bob: '),
            'name-in-par': ('[Alice]: ', '[Bob]: '),
            'dash': ('-', '-'),
            'number': ('1: ', '2: ')}
    valid_ending = ['.', '!', '?', '\'']
    
    assert wrap_type in types, "Unknown wrapping"
    
    if(insert_intro):
        output += "<|endoftext|>"#This is the conversation between 2 people."
        
    for i, msg in enumerate(m_list):
        output += types[wrap_type][i%2]
        output += msg
        if((check_end_punct) and (msg[-1] not in valid_ending)):
            output += '.'
        output += '\n'        
            
    #output += '\n'
    output += types[wrap_type][(i+1)%2]
    return output

def init_model(seed=0, model_name_or_path='gpt2'):
    '''
    Parameters:
    ----------
    seed : int
        seed number for different ramdomizers
    model_name_or_path : string, optional
        either model name for existing model or path for trained model (not realised yet)
    '''
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = GPT2Tokenizer.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load("../gpt2_model_3200.pth"))
    model = model.module
    
    model.to(device)
    model.eval()
    return model, enc, device

def model_forward(input_text, *model_params, length=-1, top_k=0, temperature=1.0):
    '''
    Parameters:
    ----------
    input_text : string
        input text for sampling
    *model_params : tuple
        (model, enc, device) output of 'init_model' function
    length : int, optional
        length of generated sample I guess (!!not sure!!)
    top_k : int, optional
        to generate k most probable samples (!!not sure!!)
    temperature: float, optional
        parameter of sampling algorithm
    '''

    model, enc, device = model_params
    if length == -1:
        length = model.config.n_ctx // 2
    elif length > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)
        
    context_tokens = []
    context_tokens = enc.encode(input_text)
    context_tokens = [50256, 220] + context_tokens
    print("Input tokens")
    print(context_tokens)
    
    out = sample_sequence(
        model=model, length=length,
        context=context_tokens,
        start_token=None,
        batch_size=1,
        temperature=temperature, top_k=top_k, device=device)

    print("Out Tokens") 
    print(out)    
    out = out[:, len(context_tokens):].tolist()
    output_text = enc.decode(out[0])
    return output_text

    
def produce_answer(user_input, prev_msgs, *model_params, **wrap_params):
    '''
    Parameters:
    ----------
    user_input : string
        user's message
    prev_msgs : list
        list of previous messages in conversation
    *model_params : tuple
        (model, enc, device) output of 'init_model' function
    **wrap_parameters : dict
        parametrs for 'wrap_message_list' function like `wrap_type`    
    '''
    prev_msgs.append(user_input)
    input_text = wrap_message_list(prev_msgs, **wrap_params)
    print("Model input:\n")
    print(input_text)
    sampled_answer = model_forward(input_text, *model_params)
    print("All sampled:\n")
    print(sampled_answer) 
    print("\n\n")
    answer = sampled_answer.split('\n')[0] ### If <end of text. -> send ...
    answer = answer.replace(u'\xa0', u'') ### FIX THIS
    prev_msgs.append(answer)
    return answer

def main():
    model, enc, device = init_model(seed=42)
    messages = []
    produce_answer("Hi! Do you have any hobbies", messages, model, enc, device, insert_intro=False, wrap_type='name')

if __name__ == '__main__':
    main()