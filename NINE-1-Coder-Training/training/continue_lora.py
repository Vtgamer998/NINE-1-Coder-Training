"""Continue LoRA training without changing base weights or tokenizer."""
import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from nine.model import NINE1, NINEConfig
from nine.finetune import add_lora, InstructDataset
from nine.tokenizer import BPETokenizer


def read_pt(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_atomic(path, state):
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, suffix='.pt', delete=False) as f:
        tmp = f.name
    try:
        torch.save(state, tmp)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@torch.no_grad()
def validation(model, loader, device):
    model.eval()
    total, count = 0., 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if (y != -100).sum().item() == 0:
            continue
        _, loss, _ = model(x, y)
        if torch.isfinite(loss):
            n = (y != -100).sum().item()
            total += loss.item() * n
            count += n
    model.train()
    return total / count if count else float('nan')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='assets/nine1-base.pt')
    p.add_argument('--adapter', default='assets/nine1-instruct.pt')
    p.add_argument('--tok', default='assets/nine1-tok.json')
    p.add_argument('--data', default='datasets/coder_ptbr.jsonl')
    p.add_argument('--out', default='checkpoints/coder-v2-resume.pt')
    p.add_argument('--export', default='checkpoints/nine1-coder-v2-lora.pt')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--save-every', type=int, default=25)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if a.steps < 1 or a.batch_size < 1 or a.save_every < 1:
        p.error('steps, batch-size e save-every devem ser positivos')
    random.seed(1337)
    torch.manual_seed(1337)
    base = read_pt(a.base)
    old = read_pt(a.adapter)
    cfg = NINEConfig.from_dict(base['cfg'])
    if NINEConfig.from_dict(old['cfg']) != cfg:
        raise ValueError('Configuração do LoRA difere da base')
    tok = BPETokenizer.load(a.tok)
    if tok.vocab_size != cfg.vocab_size:
        raise ValueError('Tamanho configurado do tokenizer difere do modelo')
    if not tok.vocab:
        raise ValueError('Tokenizer vazio')
    if min(tok.vocab) < 0 or max(tok.vocab) >= cfg.vocab_size:
        raise ValueError('IDs do tokenizer fora do vocabulário do modelo')
    model = NINE1(cfg)
    model.load_state_dict(base['model'], strict=True)
    add_lora(model, r=int(old['r']), alpha=int(old['alpha']), target=old.get('lora_target', 'qkv'))
    incompatible = model.load_state_dict(old['lora'], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError('Chaves LoRA incompatíveis: ' + str(incompatible.unexpected_keys[:8]))
    for name, param in model.named_parameters():
        param.requires_grad_('lora_' in name)
    model.to(a.device)
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=a.lr)
    dataset = InstructDataset(a.data, block_size=cfg.block_size, tokenizer=tok)
    if len(dataset) < 2:
        raise ValueError('Precisa de ao menos dois exemplos')
    n_val = max(1, len(dataset) // 10)
    train, val = random_split(dataset, [len(dataset)-n_val, n_val], generator=torch.Generator().manual_seed(42))
    # Deterministic sampler per step: loader shuffling is avoided for reproducible resume.
    val_loader = DataLoader(val, batch_size=a.batch_size, collate_fn=InstructDataset.collate_fn)
    start = 0
    fingerprint = {'base': sha(a.base), 'adapter': sha(a.adapter), 'tok': sha(a.tok), 'data': sha(a.data)}
    if a.resume:
        state = read_pt(a.out)
        if state['fingerprint'] != fingerprint:
            raise ValueError('Arquivos base/tokenizer/dataset mudaram desde o checkpoint')
        model.load_state_dict(state['model'], strict=True)
        optim.load_state_dict(state['optimizer'])
        start = state['step']
        torch.set_rng_state(state['torch_rng'])
        random.setstate(state['python_rng'])
        if a.device.startswith('cuda') and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        print('Retomando do passo', start)
    print('Validation loss antes:', validation(model, val_loader, a.device))
    model.train()
    for step in range(start+1, start+a.steps+1):
        indices = [random.randrange(len(train)) for _ in range(a.batch_size)]
        x, y = InstructDataset.collate_fn([train[i] for i in indices])
        x, y = x.to(a.device), y.to(a.device)
        if not (y != -100).any():
            raise RuntimeError('Exemplo sem tokens de resposta; revise dataset ou block_size')
        optim.zero_grad(set_to_none=True)
        _, loss, _ = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError('Loss não finita')
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optim.step()
        if step % 10 == 0:
            print(f'passo={step} loss={loss.item():.4f}', flush=True)
        if step % a.save_every == 0 or step == start + a.steps:
            save_atomic(a.out, {
                'model': model.state_dict(), 'optimizer': optim.state_dict(), 'step': step,
                'fingerprint': fingerprint, 'torch_rng': torch.get_rng_state(),
                'python_rng': random.getstate(),
                'cuda_rng': torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else None,
            })
            print('Salvo:', a.out, flush=True)
    print('Validation loss depois:', validation(model, val_loader, a.device))
    lora = {k: v.detach().cpu() for k, v in model.state_dict().items() if 'lora_' in k}
    save_atomic(a.export, {'lora': lora, 'cfg': base['cfg'], 'base': a.base,
                           'r': old['r'], 'alpha': old['alpha'],
                           'lora_target': old.get('lora_target', 'qkv')})
    print('LoRA exportado:', a.export)


if __name__ == '__main__':
    main()
