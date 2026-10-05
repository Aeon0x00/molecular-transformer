import os
import argparse
import torch
import torch.nn.functional as F
from model_decoder import build_decoder_transformer
from validate import validate_smiles
from Tokenizer import SMILES_tokenizator

def generate_molecule(model, tokenizer, device, max_len=128, temperature=1.0, top_k=50):
    sos = tokenizer.vocab['[SOS]']
    eos = tokenizer.vocab['[EOS]']
    pad = tokenizer.vocab['[PAD]']
    gen = [sos]

    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([gen]).to(device)
            seq_len = x.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(device)

            logits = model(x, mask)[:, -1, :]
            logits = logits / max(1e-5, temperature)

            k = min(top_k, logits.size(-1))
            vals, idxs = torch.topk(logits, k)
            logits = torch.full_like(logits, float('-inf')).scatter_(1, idxs, vals)

            tok = torch.multinomial(F.softmax(logits, -1), 1).item()
            if tok in (eos, pad):
                break
            gen.append(tok)

    return tokenizer.decode(gen[1:])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model_logged.pt')
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.checkpoint):
        print(f"File not found: {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    vocab = checkpoint.get('tokenizer') or checkpoint.get('tokenizer_vocab')
    tokenizer = SMILES_tokenizator(vocab=vocab)
    config = checkpoint['config']

    model = build_decoder_transformer(
        config['vocab_size'],
        d_model=config.get('d_model', 512),
        n_layers=config.get('n_layers', 8),
        heads=config.get('n_heads', 8),
        max_len=args.max_len
    ).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    valid_count = 0
    for i in range(args.num):
        smiles = generate_molecule(model, tokenizer, device, args.max_len, args.temp, args.top_k)
        if args.validate:
            ok = validate_smiles(smiles)
            valid_count += int(ok)
            print(f"{i+1}. {smiles} - {'valid' if ok else 'invalid'}")
        else:
            print(f"{i+1}. {smiles}")

    if args.validate and args.num > 0:
        print(f"\nValid: {valid_count}/{args.num} ({100*valid_count/args.num:.1f}%)")

if __name__ == '__main__':
    main()