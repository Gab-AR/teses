"""
08_token_embeddings.py

Cria a camada de token embeddings com torch.nn.Embedding(vocab_size, output_dim),
usando vocab_size=50257 (vocabulario do GPT-2) e output_dim=256.

Mostra:
  - a matriz de pesos inteira (.weight) - cuidado, e GRANDE (50257 x 256 =
    ~12.8 milhoes de numeros), entao so imprimimos um resumo + uma fatia
  - os embeddings gerados especificamente para os token ids de "Raimundo Moura"

Dependencias:
    pip install --break-system-packages torch tiktoken

Uso:
    python 08_token_embeddings.py
"""

import tiktoken
import torch


VOCAB_SIZE = 50257
OUTPUT_DIM = 256
SEED = 123  # fixa a semente para reprodutibilidade dos pesos iniciais


def main():
    torch.manual_seed(SEED)

    # --- Cria a camada de embedding ---
    # torch.nn.Embedding eh essencialmente uma tabela de consulta (lookup table):
    # uma matriz de shape (vocab_size, output_dim), onde a linha i eh o vetor
    # (embedding) do token de id i. Os valores iniciais sao aleatorios, e
    # serao ajustados durante o treinamento do modelo.
    camada_embedding = torch.nn.Embedding(VOCAB_SIZE, OUTPUT_DIM)

    print(f"Camada de embedding criada: torch.nn.Embedding({VOCAB_SIZE}, {OUTPUT_DIM})")
    print(f"Shape da matriz de pesos (.weight): {camada_embedding.weight.shape}")
    print(f"Total de parametros nessa camada: {camada_embedding.weight.numel():,}\n")

    # --- Imprime os embeddings gerados para TODO o vocabulario ---
    # Atencao: sao 50257 x 256 = ~12.86 milhoes de numeros. Imprimir tudo no
    # terminal e inviavel de ler, entao mostramos o tensor resumido (o proprio
    # PyTorch trunca a exibicao) e depois uma fatia pequena para inspecao real.
    print("--- Matriz de pesos completa (.weight), exibicao resumida pelo PyTorch ---")
    print(camada_embedding.weight)

    print("\n--- Fatia legivel: embedding dos primeiros 3 tokens do vocabulario ---")
    print(camada_embedding.weight[:3])

    # --- Embeddings para "Raimundo Moura" ---
    tokenizador = tiktoken.get_encoding("gpt2")
    texto = "Raimundo Moura"
    token_ids = tokenizador.encode(texto)
    print(f'\nTexto: "{texto}"')
    print(f"Token ids: {token_ids}")

    tensor_ids = torch.tensor(token_ids)
    embeddings_texto = camada_embedding(tensor_ids)

    print(f"\nShape dos embeddings gerados: {embeddings_texto.shape}  (num_tokens x output_dim)")
    print("Embedding de cada token:\n")
    for tok_id, vetor in zip(token_ids, embeddings_texto):
        pedaco = tokenizador.decode([tok_id])
        print(f'  token_id={tok_id:6d}  ("{pedaco}")')
        print(f"    primeiros 8 valores do vetor de {OUTPUT_DIM} dimensoes: {vetor[:8].tolist()}")

    print(
        "\nObs: esses vetores sao aleatorios agora (a camada acabou de ser"
        " criada, ainda nao foi treinada). Durante o treinamento do modelo,"
        " esses valores vao se ajustar para capturar significado semantico."
    )


if __name__ == "__main__":
    main()
