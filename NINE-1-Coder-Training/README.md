# NINE-1 Coder Training

Projeto separado para continuar o ajuste fino **LoRA** da NINE-1 original. Não reinicia os pesos do modelo nem modifica o tokenizador. O checkpoint completo salva pesos do adaptador, estado do otimizador, passo e estados aleatórios. O checkpoint antigo só contém pesos; portanto, na primeira execução é um *warm start*, não uma retomada exata do otimizador antigo.

## No GitHub Codespaces

```bash
git clone https://github.com/Vtgamer998/nine-1.git upstream-nine
python -m pip install -r requirements.txt
mkdir -p assets
```

Copie os três arquivos de `modelo_treinado (1).zip` para `assets/`: `nine1-base.pt`, `nine1-instruct.pt` e `nine1-tok.json`. **Não faça commit desses pesos** (já estão no `.gitignore`).

```bash
PYTHONPATH=upstream-nine:. python -m training.continue_lora --steps 100 --device cpu
```

Para retomar do checkpoint criado pelo comando anterior:

```bash
PYTHONPATH=upstream-nine:. python -m training.continue_lora --resume --steps 100 --device cpu
```

Se o Codespace tiver CUDA, troque `--device cpu` por `--device cuda`.

**Importante:** o projeto original pode mudar. Este script foi preparado para `nine.model`, `nine.finetune` e `nine.tokenizer` do repositório atual. Faça uma cópia de segurança do modelo antes de treinar. O dataset incluído é pequeno e serve para validar o pipeline; não garante melhor capacidade de programação. A avaliação calcula loss de validação, não execução correta de código. Treinar em CPU pode ser lento.
