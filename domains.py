from torch import load
import os

def load_domains(
    one_embed_dir = os.path.join('part_1_vit_embeds'),
    two_embed_dir = os.path.join('part_2_vit_embeds')
):
    one_train_dir = os.path.join('dataset', 'part_one_dataset', 'train_data')
    one_eval_dir = os.path.join('dataset', 'part_one_dataset', 'eval_data')
    two_train_dir = os.path.join('dataset', 'part_two_dataset', 'train_data')
    two_eval_dir = os.path.join('dataset', 'part_two_dataset', 'eval_data')
    
    domains = [{} for _ in range(20)]
    eval_domains = [{} for _ in range(20)]

    for j in range(10):
        
        train_path = os.path.join(one_train_dir, f'{j+1}_train_data.tar.pth')
        t = load(train_path, weights_only = False)
        
        domains[j]['labels'] = t['targets'] if 'targets' in t else None
        domains[j]['features'] = load(os.path.join(one_embed_dir,f'train_embeds_{j+1}.pt'), weights_only = False)
        
        eval_path = os.path.join(one_eval_dir, f'{j+1}_eval_data.tar.pth')
        t = load(eval_path, weights_only = False)

        eval_domains[j]['labels'] = t['targets']
        eval_domains[j]['features'] = load(os.path.join(one_embed_dir,f'eval_embeds_{j+1}.pt'), weights_only = False)
        
        train_path = os.path.join(two_train_dir, f'{j+1}_train_data.tar.pth')
        t = load(train_path, weights_only = False)
        
        domains[j+10]['labels'] = t['targets'] if 'targets' in t else None
        domains[j+10]['features'] = load(os.path.join(two_embed_dir,f'train_embeds_{j+1}.pt'), weights_only = False)
        
        eval_path = os.path.join(two_eval_dir, f'{j+1}_eval_data.tar.pth')
        t = load(eval_path, weights_only = False)
        
        eval_domains[j+10]['labels'] = t['targets']
        eval_domains[j+10]['features'] = load(os.path.join(two_embed_dir,f'eval_embeds_{j+1}.pt'), weights_only = False)
        
    return domains, eval_domains