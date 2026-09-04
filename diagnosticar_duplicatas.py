"""
diagnosticar_duplicatas.py

Mostra os titulos normalizados que mais aparecem em possiveis_duplicatas.csv,
para confirmar se o problema e nome de instituicao sendo capturado como
titulo (ou outro padrao de falso positivo).

Uso:
    python diagnosticar_duplicatas.py --curated-dir teses/curated
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-dir", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    caminho = Path(args.curated_dir) / "possiveis_duplicatas.csv"
    df = pd.read_csv(caminho)

    contagem = df["titulo_normalizado"].value_counts()

    print(f"Total de grupos: {df['titulo_normalizado'].nunique()}")
    print(f"Total de registros envolvidos: {len(df)}\n")
    print(f"Top {args.top_n} titulos normalizados mais frequentes (grupos suspeitos):\n")
    for titulo, n in contagem.head(args.top_n).items():
        print(f"  [{n:4d}] {titulo[:90]}")

    grandes = contagem[contagem > 5]
    print(f"\nGrupos com mais de 5 membros (quase certamente falso positivo): {len(grandes)}")
    print(f"Registros nesses grupos grandes: {grandes.sum()}")


if __name__ == "__main__":
    main()
