"""
09_embeddings_posicionais.py

Cria embeddings posicionais absolutos para uma janela de contexto de
max_length=6 tokens, usando os valores sequenciais pedidos no enunciado:
  posicao 1: 1.1, 1.2, 1.3, ..., 1.256   (256 = output_dim)
  posicao 2: 2.1, 2.2, 2.3, ..., 2.256
  ...
  posicao 6: 6.1, 6.2, 6.3, ..., 6.256

Depois soma esses embeddings posicionais aos token embeddings (gerados no
script 08) para obter os "input embeddings" finais, que sao o que
efetivamente entra no modelo tipo GPT.

IMPORTANTE - nota tecnica:
Este esquema de "1.1, 1.2, ..." e uma forma DIDATICA pedida no enunciado
para deixar visualmente claro qual token esta em qual posicao. Nao e o
esquema usado de fato no GPT-2 real: la, os embeddings posicionais tambem
vem de uma torch.nn.Embedding(max_length, output_dim) com pesos treinaveis
(aleatorios no inicio, ajustados durante o treino), exatamente como os
token embeddings. Ambas as abordagens estao implementadas abaixo, para
fins de comparacao.

Dependencias:
    pip install --break-system-packages torch tiktoken

Uso:
    python 09_embeddings_posicionais.py
"""

import tiktoken
import torch


VOCAB_SIZE = 50257
OUTPUT_DIM = 256
MAX_LENGTH = 6  # janela de contexto pedida no enunciado
SEED = 123


def criar_embeddings_posicionais_didaticos(max_length: int, output_dim: int) -> torch.Tensor:
    """
    Gera a matriz de embeddings posicionais no formato pedido no enunciado:
    linha 1 = [1.1, 1.2, ..., 1.output_dim]
    linha 2 = [2.1, 2.2, ..., 2.output_dim]
    etc.

    Isso NAO e como o GPT-2 real gera embeddings posicionais (la sao pesos
    aprendidos), mas serve para visualizar claramente, de forma didatica,
    que cada posicao da janela tem seu proprio vetor.
    """
    matriz = torch.zeros(max_length, output_dim)
    for posicao in range(max_length):
        numero_posicao = posicao + 1  # comeca em 1, nao em 0
        for dim in range(output_dim):
            numero_dimensao = dim + 1
            # Monta o valor "posicao.dimensao", ex: posicao=1, dim=23 -> 1.23
            valor = float(f"{numero_posicao}.{numero_dimensao}")
            matriz[posicao, dim] = valor
    return matriz


def criar_embeddings_posicionais_treinaveis(max_length: int, output_dim: int, seed: int = SEED) -> torch.nn.Embedding:
    """
    Forma real usada em implementacoes do GPT-2: uma camada de embedding
    treinavel, com uma linha (vetor) por posicao na janela de contexto.
    """
    torch.manual_seed(seed)
    return torch.nn.Embedding(max_length, output_dim)


def main():
    torch.manual_seed(SEED)

    # --- 1) Token embeddings (igual ao script 08) para "Raimundo Moura" ---
    tokenizador = tiktoken.get_encoding("gpt2")
    texto = "Raimundo Moura"
    token_ids = tokenizador.encode(texto)
    print(f'Texto: "{texto}"')
    print(f"Token ids: {token_ids}  (total: {len(token_ids)} tokens)\n")

    camada_token_embedding = torch.nn.Embedding(VOCAB_SIZE, OUTPUT_DIM)
    tensor_ids = torch.tensor(token_ids)
    token_embeddings = camada_token_embedding(tensor_ids)
    print(f"Shape dos token embeddings: {token_embeddings.shape}  (num_tokens x output_dim)")

    # --- 2) Embeddings posicionais didaticos (1.1, 1.2, ... pedido no enunciado) ---
    embeddings_posicionais = criar_embeddings_posicionais_didaticos(MAX_LENGTH, OUTPUT_DIM)
    print(f"\nShape dos embeddings posicionais didaticos: {embeddings_posicionais.shape}  (max_length x output_dim)")
    print("Primeiros 6 valores de cada posicao (para conferir o padrao 1.1, 1.2, ...):")
    for i in range(MAX_LENGTH):
        print(f"  posicao {i+1}: {embeddings_posicionais[i, :6].tolist()}")

    # --- 3) Soma token embeddings + embeddings posicionais ---
    # "Raimundo Moura" tem 5 tokens (menos que MAX_LENGTH=6), entao usamos
    # so as primeiras 5 linhas dos embeddings posicionais - uma posicao por
    # token, na ordem em que aparecem na janela de contexto.
    n_tokens = token_embeddings.shape[0]
    posicionais_recortados = embeddings_posicionais[:n_tokens]

    input_embeddings = token_embeddings + posicionais_recortados

    print(f"\n--- Soma: token_embeddings + embeddings_posicionais ---")
    print(f"Shape final dos input embeddings: {input_embeddings.shape}")
    print("\nPrimeiros 6 valores de cada token, apos a soma:")
    for tok_id, vetor in zip(token_ids, input_embeddings):
        pedaco = tokenizador.decode([tok_id])
        print(f'  token_id={tok_id:6d}  ("{pedaco}")  ->  {vetor[:6].tolist()}')

    # --- 4) Comparacao: como seria com embeddings posicionais TREINAVEIS (GPT-2 real) ---
    print("\n\n=== Comparacao: embeddings posicionais treinaveis (forma real do GPT-2) ===")
    camada_posicional_treinavel = criar_embeddings_posicionais_treinaveis(MAX_LENGTH, OUTPUT_DIM)
    posicoes = torch.arange(n_tokens)  # [0, 1, 2, 3, 4] para 5 tokens
    embeddings_pos_treinaveis = camada_posicional_treinavel(posicoes)

    input_embeddings_real = token_embeddings + embeddings_pos_treinaveis
    print(f"Shape dos input embeddings (versao treinavel): {input_embeddings_real.shape}")
    print(
        "\nNessa versao, os valores posicionais comecam aleatorios (nao"
        " seguem o padrao 1.1, 1.2, ...) e sao ajustados durante o"
        " treinamento do modelo, assim como os token embeddings."
    )


if __name__ == "__main__":
    main()
