"""
07_dataset_dataloader.py

Le o arquivo .txt gerado na etapa anterior (ex: amostra_teses.txt), tokeniza
com tiktoken (gpt2), e usa janela deslizante para gerar pares input-target:
para cada posicao, o input e uma janela de max_length tokens, e o target e
a mesma janela deslocada 1 token para frente (proxima palavra a prever).

Cria:
  - DatasetTexto: Dataset customizado do PyTorch
  - criar_dataloader(): monta o DataLoader com embaralhamento e batches

Dependencias:
    pip install --break-system-packages tiktoken torch

Uso:
    python 07_dataset_dataloader.py --arquivo-txt teses/curated/amostra_teses.txt
"""

import argparse
from pathlib import Path

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class DatasetTexto(Dataset):
    """
    Dataset customizado: recebe um texto bruto ja tokenizado (lista de token
    ids) e gera pares (input, target) via janela deslizante.

    Para cada posicao i (de stride em stride), pega:
      input  = tokens[i : i + max_length]
      target = tokens[i+1 : i + max_length + 1]   (mesma janela, deslocada em 1)

    'stride' controla a sobreposicao entre janelas consecutivas:
      stride == max_length  -> nenhuma sobreposicao
      stride < max_length   -> janelas se sobrepoem (mais amostras, mais redundancia)
    """

    def __init__(self, texto: str, tokenizador, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizador.encode(texto, allowed_special={"<|endoftext|>"})
        print(f"Total de tokens no texto: {len(token_ids)}")

        for i in range(0, len(token_ids) - max_length, stride):
            janela_input = token_ids[i : i + max_length]
            janela_target = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(janela_input, dtype=torch.long))
            self.target_ids.append(torch.tensor(janela_target, dtype=torch.long))

        print(f"Total de janelas (amostras) geradas: {len(self.input_ids)}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def criar_dataloader(
    texto: str,
    max_length: int = 6,
    stride: int = 6,
    batch_size: int = 4,
    embaralhar: bool = True,
    descartar_ultimo_incompleto: bool = True,
    num_workers: int = 0,
):
    """Monta o tokenizador GPT-2, o Dataset e o DataLoader prontos para uso."""
    tokenizador = tiktoken.get_encoding("gpt2")
    dataset = DatasetTexto(texto, tokenizador, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=embaralhar,
        drop_last=descartar_ultimo_incompleto,
        num_workers=num_workers,
    )
    return dataloader, tokenizador


def main():
    parser = argparse.ArgumentParser(description="Gera pares input-target via janela deslizante")
    parser.add_argument("--arquivo-txt", required=True, help="Arquivo .txt de entrada (ex: amostra_teses.txt)")
    parser.add_argument("--max-length", type=int, default=6, help="Tamanho da janela de contexto (numero de tokens)")
    parser.add_argument("--stride", type=int, default=6, help="Passo da janela deslizante")
    parser.add_argument("--batch-size", type=int, default=4, help="Tamanho do batch")
    parser.add_argument("--max-chars", type=int, default=50_000, help="Le so os primeiros N caracteres do arquivo (0 = arquivo inteiro)")
    args = parser.parse_args()

    caminho = Path(args.arquivo_txt)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho}")

    texto = caminho.read_text(encoding="utf-8", errors="ignore")
    if args.max_chars > 0:
        texto = texto[: args.max_chars]
        print(f"Usando os primeiros {args.max_chars} caracteres do arquivo (--max-chars)")

    dataloader, tokenizador = criar_dataloader(
        texto,
        max_length=args.max_length,
        stride=args.stride,
        batch_size=args.batch_size,
    )

    print(f"\nParametros: max_length={args.max_length}  stride={args.stride}  batch_size={args.batch_size}")
    print(f"Numero de batches no DataLoader: {len(dataloader)}\n")

    # Mostra o primeiro batch como exemplo
    primeiro_batch_inputs, primeiro_batch_targets = next(iter(dataloader))
    print("--- Primeiro batch ---")
    print(f"Shape dos inputs:  {primeiro_batch_inputs.shape}   (batch_size x max_length)")
    print(f"Shape dos targets: {primeiro_batch_targets.shape}")
    print(f"\nInputs (token ids):\n{primeiro_batch_inputs}")
    print(f"\nTargets (token ids):\n{primeiro_batch_targets}")

    print("\n--- Decodificando o primeiro exemplo do batch, para conferir ---")
    exemplo_input = primeiro_batch_inputs[0].tolist()
    exemplo_target = primeiro_batch_targets[0].tolist()
    print(f"Input:  {exemplo_input}  ->  \"{tokenizador.decode(exemplo_input)}\"")
    print(f"Target: {exemplo_target}  ->  \"{tokenizador.decode(exemplo_target)}\"")
    print(
        "\nRepare que o target e o input deslocado em 1 posicao: em cada"
        " passo, o modelo aprende a prever o PROXIMO token dado o contexto"
        " anterior."
    )


if __name__ == "__main__":
    main()
