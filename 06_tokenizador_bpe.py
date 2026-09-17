"""
06_tokenizador_bpe.py

Carrega o esquema de tokenizacao Byte Pair Encoding (BPE) usado no GPT-2,
via tiktoken. Confirma o tamanho do vocabulario (deve ser 50257) e testa
encode()/decode() em alguns textos, incluindo "Raimundo Moura".

Dependencias:
    pip install --break-system-packages tiktoken

Uso:
    python 06_tokenizador_bpe.py
"""

import tiktoken


def main():
    # "gpt2" carrega o esquema de tokenizacao BPE original do GPT-2.
    # Outros LLMs da OpenAI usam esquemas diferentes (ex: "cl100k_base" para
    # GPT-3.5/4), mas o pedido aqui e especificamente o do GPT-2.
    tokenizador = tiktoken.get_encoding("gpt2")

    tamanho_vocab = tokenizador.n_vocab
    print(f"Esquema de tokenizacao carregado: gpt2")
    print(f"Tamanho do vocabulario: {tamanho_vocab}")
    assert tamanho_vocab == 50257, "Tamanho de vocabulario inesperado!"
    print("Confirmado: vocabulario tem exatamente 50.257 tokens.\n")

    # --- Teste principal pedido ---
    texto_teste = "Raimundo Moura"
    tokens = tokenizador.encode(texto_teste)
    print(f'Texto: "{texto_teste}"')
    print(f"Tokens gerados: {tokens}")
    print(f"Esperado:       [49, 1385, 41204, 49902, 403]")
    print(f"Bate com o esperado? {tokens == [49, 1385, 41204, 49902, 403]}\n")

    # Decodificando de volta, e tambem token a token (para ver a quebra BPE)
    texto_decodificado = tokenizador.decode(tokens)
    print(f"Decodificado de volta: \"{texto_decodificado}\"")
    print("Quebra token a token:")
    for tok_id in tokens:
        pedaco = tokenizador.decode([tok_id])
        print(f"  id={tok_id:6d}  ->  \"{pedaco}\"")

    # --- Testes livres com outros 3 textos ---
    print("\n--- Testes livres ---")
    outros_textos = [
        "Universidade Federal do Piaui",
        "Inteligencia artificial e aprendizado de maquina",
        "O quick brown fox jumps over the lazy dog",
    ]
    for texto in outros_textos:
        tokens = tokenizador.encode(texto)
        decodificado = tokenizador.decode(tokens)
        print(f'\nTexto original:  "{texto}"')
        print(f"Tokens ({len(tokens)}):     {tokens}")
        print(f"Decodificado:     \"{decodificado}\"")
        print(f"Round-trip OK?    {decodificado == texto}")


if __name__ == "__main__":
    main()
