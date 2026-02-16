import torch
import re
from dataclasses import dataclass
from datasets import load_dataset
# from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, random_split, DataLoader
from tqdm import tqdm

class LogisticRegression(torch.nn.Module):
    def __init__(self, in_feature_size) -> None:
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(in_features=in_feature_size, out_features=512)
        self.act_fn = torch.nn.ReLU()
        self.linear_1 = torch.nn.Linear(in_features=512, out_features=2)
    
    def forward(self, x):
        y_pred = self.linear(x)
        y_pred = self.act_fn(y_pred)
        y_pred = self.linear_1(y_pred)
        return y_pred

class ProbePoolDataset(Dataset):
    def __init__(self, dataset_name) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.data = torch.load(self.dataset_name)
    
    def __len__(self):
        return len(self.data)
    
    def _set_up_pca(self):
        pass

    def __getitem__(self, index):
        data = self.data[f'sample_{index}']
        
        pooled_layers = []
        for key in sorted(data.keys()):
            if key.startswith("layer_"):
                layer_tensor = data[key]  
                pooled = layer_tensor.mean(dim=1)
                pooled = pooled.squeeze(0)
                pooled_layers.append(pooled)

        pooled_layers = torch.stack(pooled_layers)
        # print(pooled_layers.shape)
        pooled_vector = pooled_layers.mean(dim=0)
        # print(pooled_vector.shape)

        label = data['label']

        return {
            'layer_wise_data': pooled_vector,
            'label': label
        }

def formatting_prompt_func(examples):
    prompt = examples['instruction']
    input_ = examples['input']
    sys_prompt="""
    You are an instruction following agent.
    """
    texts = []

    for input, output in zip(prompt, input_):
        text = [
            {'role': 'system', 'content': f'{sys_prompt}'},
            {'role': 'user', 'content': f'## Instruction: {input}\n ## Input: {output}'},
        ]
        texts.append(text)
    
    return {"text_prep" : texts}    

def make_probe_data(model, tokenizer, save_pt_file='default.pt'):
    dataset_harm = load_dataset('gjyotin305/activation_data_harm', split='train')
    dataset_harm = dataset_harm.map(formatting_prompt_func, batched=True, num_proc=3)
    output_dict = {}
    pbar = tqdm(total=len(dataset_harm), desc="Extracting Hidden States")

    for idx_d, x in enumerate(dataset_harm):
        tokenized_in  = tokenizer.apply_chat_template(
            x['text_prep'],
            tokenize=True,
            return_tensors='pt',
            add_generation_prompt=True,
        )
        
        output = model.forward(tokenized_in.to('cuda'), output_hidden_states=True)
        out = {}

        for idx, hidden_state in enumerate(output.hidden_states):
            out[f'layer_{idx}'] = hidden_state.detach().cpu()

        out['label'] = x['label']
        output_dict[f'sample_{idx_d}'] = out
        pbar.set_postfix({
            'len_out': len(output_dict)
        })
        pbar.update(1)
    
    torch.save(output_dict, save_pt_file)
    return save_pt_file, True


def train_probe_data(save_pt_file, in_feature_size):
    probe_dataset = ProbePoolDataset(dataset_name=save_pt_file)

    train_size = int(0.8 * len(probe_dataset))
    test_size = len(probe_dataset) - train_size

    train_dataset, test_dataset = random_split(
        probe_dataset,
        [train_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    full_loader = DataLoader(
        probe_dataset,
        batch_size=32,
        shuffle=False
    )

    training_config = {
        'lr': 1e-3,
        'epochs': 100
    }

    model = LogisticRegression(
        in_feature_size=in_feature_size
    ).to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config['lr'])
    loss_module = torch.nn.CrossEntropyLoss()

    best_acc = 0
    pbar = tqdm(total=training_config['epochs'], desc="Training Probe")

    for epoch in range(training_config['epochs']):
        total_train_loss = 0
        for item in train_loader:
            acts = item['layer_wise_data']
            y_pred = model.forward(acts.to('cuda'))
            loss = loss_module(y_pred, item['label'].to('cuda'))
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for item in test_loader:
                acts = item["layer_wise_data"].to('cuda')
                labels = item["label"].to('cuda')

                logits = model(acts)

                # Accuracy
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        eval_acc = correct / total
        if eval_acc > best_acc:
            best_acc = eval_acc
            full_dict = {
                'epoch': epoch,
                'acc': eval_acc,
                'model': model.state_dict()
            }
            torch.save(full_dict, 'best_mod.pt')
        pbar.update(1)
        pbar.set_postfix(
            {
                'eval_acc': eval_acc,
                'best_acc': best_acc
            }
        )

    del model
    # print('Full Evaluation with best model')
    correct = 0
    total = 0

    model = LogisticRegression(
        in_feature_size=in_feature_size
    )
    checkpoint = torch.load('best_mod.pt', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model = model.to('cuda')
    with torch.no_grad():
        for item in full_loader:
            acts = item["layer_wise_data"].to('cuda')
            labels = item["label"].to('cuda')

            logits = model(acts)

            # Accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    eval_acc = correct / total
    print(f'Full Eval Accuracy: {eval_acc}')


def find_subsequence(full_ids, sub_ids):
    """
    Find start index where sub_ids occurs inside full_ids.
    Returns None if not found.
    """
    n = len(sub_ids)

    for i in range(len(full_ids) - n + 1):
        if torch.equal(full_ids[i:i+n], sub_ids):
            return i

    return None


def split_by_assistant_prefix(input_ids, tokenizer, assistant_prefix):
    """
    Split input_ids into:

        prompt_ids    = tokens BEFORE assistant prefix
        assistant_ids = tokens FROM assistant prefix onward

    Requires only:
        - input_ids tensor
        - tokenizer
    """

    # Ensure 1D tensor
    if input_ids.dim() == 2:
        input_ids = input_ids[0]

    assistant_prefix_str = assistant_prefix

    prefix_ids = tokenizer(
        assistant_prefix_str,
        add_special_tokens=False
    ).input_ids

    prefix_ids = torch.tensor(prefix_ids)

    start_idx = find_subsequence(input_ids, prefix_ids.to('cuda'))

    if start_idx is None:
        raise ValueError("Assistant prefix not found inside input_ids!")

    prompt_ids = input_ids[:start_idx+2]
    assistant_ids = input_ids[start_idx:]

    return {
        "assistant_prefix_str": assistant_prefix_str,
        "assistant_start": start_idx,
        "prompt_ids": prompt_ids,
        "assistant_ids": assistant_ids,
    }


def get_activations(model, input):
    out = {}
    with torch.no_grad():
        output = model.forward(input, output_hidden_states=True)
    
        for idx, hidden_state in enumerate(output.hidden_states):
            out[f'layer_{idx}'] = hidden_state

    return out

def predict_harm_score(probe, ref_model, input, tokenizer, assistant_prefix):
    new_input_ids = split_by_assistant_prefix(input_ids=input, tokenizer=tokenizer, assistant_prefix=assistant_prefix)
    # print(new_input_ids['prompt_ids'].shape)
    ref_model_activations = get_activations(model=ref_model, input=new_input_ids['prompt_ids'].unsqueeze(0))
    
    pooled_layers = []
    for key in sorted(ref_model_activations.keys()):
        if key.startswith("layer_"):
            layer_tensor = ref_model_activations[key]  
            pooled = layer_tensor.mean(dim=1)
            pooled = pooled.squeeze(0)
            pooled_layers.append(pooled)

    pooled_layers = torch.stack(pooled_layers)
    pooled_vector = pooled_layers.mean(dim=0)
    
    with torch.no_grad():
        logits = probe(pooled_vector.unsqueeze(0))

    # print(logits.shape)
    preds = torch.argmax(logits, dim=1)
    # print(preds)
    if preds[0] == 1:
        return 5.0
    else:
        return 1.0
        
