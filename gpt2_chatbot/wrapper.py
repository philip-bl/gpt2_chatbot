from model_sampler import *

import numpy as np
import torch
import torch.nn as nn
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
    types = {'name': ('Alice:', 'Bob:'),
            #'name-in-par': ('[Alice]:', '[Bob]:'),
            'dash': ('-', '-'),
            'number': ('1:', '2:')}
    valid_ending = ['.', '!', '?']
    
    assert wrap_type in types, "Unknown wrapping"
    
    if(insert_intro):
        #output += "<|endoftext|>"
        output += "This is a conversation between 2 people.\n"
        
    for i, msg in enumerate(m_list):
        output += types[wrap_type][i%2]
        output += ' '
        output += msg
        if((check_end_punct) and (msg[-1] not in valid_ending)):
            output += '.'
        output += '\n'        
            
    #output += '\n'
    output += types[wrap_type][(i+1)%2]
    
    if(types[wrap_type][0][-1] == ':'):
        conditioning = types[wrap_type][0][:-1]
    else:
        conditioning = types[wrap_type][0]
    
    return output, [conditioning]

def init_model(seed=0, model_path='gpt2'):
    '''
    Parameters:
    ----------
    seed : int
        seed number for different ramdomizers
    model_name_or_path : string, optional
        either model name for existing model or path for trained model
    '''
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model = model.module
    
    model.to(device)
    model.eval()
    return model, enc, device

def model_forward(input_text, conditioning, verbose, *model_params, length=128, top_k=10, temperature=1.0):
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
    context_tokens = [50256] + context_tokens
    
    cond_tokens = []
    for token in conditioning:
        cond_tokens += enc.encode(token)
    
    if(verbose):
        print('Detected conditioning tokens:')
        print(cond_tokens)
        print("Input tokens:")
        print(context_tokens)
    
    out = sample_sequence(
        model=model, length=length,
        context=context_tokens, cond_tokens=cond_tokens,
        start_token=None,
        batch_size=1,
        temperature=temperature, top_k=top_k, device=device)
    
    if(verbose):
        print("Out Tokens:") 
        print(out)    
    out = out[:, len(context_tokens):].tolist()
    output_text = enc.decode(out[0])
    return output_text

def split(string, delimeters):
    sentences = []
    prev_end = 0
    for i in range(1, len(string)):
        if((string[i-1] in delimeters) & (string[i] not in delimeters)):
            str_to_append = string[prev_end:i-1].strip() + string[i-1]
            sentences.append((str_to_append[0].upper() + str_to_append[1:]).replace('¿', ''))
            prev_end = i
    str_to_append = string[prev_end:-1].strip() + string[-1]
    #print("Strting to append:", '\''+str_to_append+'\'')
    sentences.append((str_to_append[0].upper() + str_to_append[1:]).replace('¿', ''))
    return sentences

def output_post_processing(input_quote, max_words):
    '''
    Parameters:
    ----------
    input_quote : string
        output of model
    max_words : integer
        maximal number of words (rounded to the end of sentence with last word) to concatenate to phrase
    '''
    valid_endings = ['.', '!', '?', '¿']
    input_quote = input_quote.replace(u'\xa0', '') # filter out "\xa0" 
    input_quote = input_quote.replace('\n', ' ') # filter out "\n" 
    
    first_endoftext = input_quote.find('<|endoftext|>') 
    if(first_endoftext != -1):
        input_quote = input_quote[:first_endoftext] # cut the string when '<|endoftext|>' found'
        
    first_Alice = input_quote.find('Alice:') 
    if(first_Alice != -1):
        input_quote = input_quote[:first_Alice] # cut the string when 'Alice:' found'

    input_quote = input_quote.replace("Bob:", '¿') # filter out "Bob: "
    #print('Partially processed: ')
    #print(input_quote+'|')
    
    sentences = split(input_quote.strip(), valid_endings) # sprit remaining string to sentences according to delimiters
    
    #print(sentences)
    
    sentences = list(filter(None, sentences)) # filter out empty strings
    sentences = list(filter(lambda x: x != '¿', sentences))  # filter out empty strings
    
    #print(sentences)

    for i, sentence in enumerate(sentences): # add periods where nessecary
#         for j in range(len(sentence)-1, 1, -1):
#             if((not sentence[j].isalnum()) & (sentence[j-1].isalnum())):
#                 left_part = sentence[:j]
#                 right_part = sentence[j:]
#                 print("\n\n!!!SPACE BETWEEN PUNCTUATION AND WORDS!!!")
#                 print("BEFORE:")
#                 print(left_part, right_part)
#                 right_part = right_part.replace(' ', '')
#                 print("AFTER:")
#                 print(left_part + right_part)
#                 sentences[i] = left_part + right_part
#                 break
        
        if(sentence[-1] not in valid_endings):
            sentences[i] += '.'
    
    word_counts = [len(s.split(' ')) for s in sentences]
    word_cum_counts = np.cumsum(np.array(word_counts)) / max_words
    sentences_to_pass = np.sum(word_cum_counts < 1.0) + 1
    
    return " ".join(sentences[:sentences_to_pass])
    
def produce_answer(user_input, prev_msgs, max_words, filter_attempts=1, top_k=10, temperature=1.0, verbose=False, *model_params, **wrap_params):
    '''
    Parameters:
    ----------
    user_input : string
        user's message

    prev_msgs : list
        list of previous messages in conversation

    max_words : integer
        number of words to generate (rounded to the end of last sentence)
    
    filter_attemps : integer
        maximum number of attempts to filter stop-words

    *model_params : tuple
        (model, enc, device) output of 'init_model' function
        
    **wrap_parameters : dict
        parametrs for 'wrap_message_list' function like `wrap_type`    
    '''



    prev_msgs.append(user_input)
    input_text, conditioning = wrap_message_list(prev_msgs, **wrap_params)
    stop_words = [' asl', ' f ', ' m ',  ' fuck', ' suck', ' dick', ' horny']
    if(verbose):
            print("Model input:\n")
            print(input_text)
    for _ in range(filter_attempts):
        sampled_answer = model_forward(input_text, conditioning, verbose, top_k=top_k, temperature=temperature, *model_params)
        if(verbose):
            print("All sampled:\n")
            print(sampled_answer) 
            print("\n\n")
            
        answer = output_post_processing(sampled_answer, max_words)

        search_list = []
        for word in stop_words:
            search_list.append(answer.lower().find(word))
        # print(search_list)
        if(np.array(search_list).sum() == -1*len(stop_words)):
            break

    prev_msgs.append(answer)
    return answer

def main():
    model, enc, device = init_model(42, "../gpt2_model_52800.pth")
    messages = []
    while(True):
        print("\n")
        input_text = input("Enter your message here: ")
        output_text = produce_answer(input_text, messages, 30, 5, 10, 1.0, True, model, enc, device, insert_intro=True, wrap_type='name')
        print(output_text)

if __name__ == '__main__':
    main()