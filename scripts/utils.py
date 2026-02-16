import tokenize
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import os
from prompts import PROMPT_SCORE

judge_client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)

def extract_content(tag, text):
    # Find the starting position of the tag
    start_idx = text.find(tag)

    # If tag is not found, return None
    if start_idx == -1:
        return None
    
    # Extract the content after the tag
    content_after_tag = text[start_idx+len(tag):].strip()
    
    # Split the content by whitespace
    parts = content_after_tag.split()
    
    if tag == "#thescore:":
        assert parts[0].isdigit()
        return int(parts[0])
    else:
        end_idx = text.find("#", start_idx + 1)
        return content_after_tag if end_idx == -1 else content_after_tag[:end_idx].strip()

def segregate_messages(text, template='qwen'):
    if template == 'qwen':
        pattern = r"<\|im_start\|>(.*?)\n(.*?)<\|im_end\|>"
    elif template == 'llama':
        pattern = r"<\|start_header_id\|>(.*?)<\|end_header_id\|>(.*?)<\|eot_id\|>"
    elif template == 'phi':
        pattern = r"<\|(system|user|assistant)\|>\s*(.*?)\s*<\|end\|>"
    matches = re.findall(pattern, text, flags=re.DOTALL)

    result = []
    for role, content in matches:
        role = role.strip()
        content = content.strip()
        result.append({"role": role, "content": content})
    
    return result

def get_user_and_assistant(text, template='qwen'):
    messages = segregate_messages(text, template=template)

    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    assistant_msgs = [m["content"] for m in messages if m["role"] == "assistant"]

    return user_msgs, assistant_msgs


def get_stream_judge_response(input, tokenizer, model_name='gpt-4.1-mini', template='qwen'):
    score_prompt = PROMPT_SCORE

    scores = []
    for text in input['input_ids']:
        str_input = tokenizer.decode(text)
        user, answer = get_user_and_assistant(str_input, template=template)
        
        try:
            judge_answer = score_prompt.format(user[0], answer[0])
            
            response = judge_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": judge_answer}
                ],
                temperature=0.1,
                max_tokens=2048
            )
            
            score = extract_content("#thescore:", response.choices[0].message.content)
            scores.append(score)
        except Exception as e:
            print(f'Unverified response {e}')
            scores.append(4)
    try:
        scores = [score for score in scores if score is not None]
        score_harm = sum(scores)/len(scores)
    except:
        print('Exception Occured, Saving Model')
        score_harm = None

    return score_harm