"""
05_amostra_e_estatisticas.py

Carrega uma amostra (ou o total) do catalogo curated + textos completos,
mostra estatisticas basicas do dataset e exporta tudo em um unico arquivo
.txt, com um delimitador claro separando cada documento.

Estatisticas exibidas:
  - numero de documentos na amostra
  - numero medio de paragrafos por documento
  - numero medio de palavras por documento
  - (extras) mediana de palavras, min/max, total de palavras no corpus

Uso:
    # Amostra aleatoria de 500 documentos
    python 05_amostra_e_estatisticas.py \
        --curated-dir teses/curated \
        --processed-dir teses/processed \
        --n-amostra 500 \
        --saida teses/curated/amostra_teses.txt

    # Dataset inteiro (sem amostragem)
    python 05_amostra_e_estatisticas.py \
        --curated-dir teses/curated \
        --processed-dir teses/processed \
        --n-amostra 0 \
        --saida teses/curated/dataset_completo.txt
"""

import argparse
import re
import statistics
import sys
from pathlib import Path

import pandas as pd


DELIMITADOR_INICIO = "===== INICIO_DOCUMENTO id={id_hash} titulo=\"{titulo}\" ano={ano} ====="
DELIMITADOR_FIM = "===== FIM_DOCUMENTO id={id_hash} ====="


def contar_paragrafos(texto: str) -> int:
    """Paragrafo = bloco de texto separado por linha em branco."""
    blocos = [b for b in re.split(r"\n\s*\n", texto) if b.strip()]
    return len(blocos)


def contar_palavras(texto: str) -> int:
    return len(texto.split())


def carregar_texto(fulltext_dir: Path, id_hash: str) -> str:
    caminho = fulltext_dir / f"{id_hash}.txt"
    if not caminho.exists():
        return ""
    return caminho.read_text(encoding="utf-8", errors="ignore")


def main():
    parser = argparse.ArgumentParser(description="Amostra + estatisticas + exportacao em txt do dataset curated")
    parser.add_argument("--curated-dir", required=True, help="Pasta curated (contem catalogo.parquet)")
    parser.add_argument("--processed-dir", required=True, help="Pasta processed (contem fulltext/)")
    parser.add_argument("--n-amostra", type=int, default=500, help="Tamanho da amostra aleatoria (0 = usa o dataset inteiro)")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatoria, para amostra reprodutivel")
    parser.add_argument("--saida", required=True, help="Caminho do .txt de saida")
    parser.add_argument(
        "--apenas-sem-flag",
        action="store_true",
        help="Se definido, amostra so entre documentos com flag_qualidade == 'ok' (exclui os marcados para revisao manual)",
    )
    args = parser.parse_args()

    curated_dir = Path(args.curated_dir).resolve()
    processed_dir = Path(args.processed_dir).resolve()
    fulltext_dir = processed_dir / "fulltext"
    caminho_catalogo = curated_dir / "catalogo.parquet"
    saida_path = Path(args.saida).resolve()
    saida_path.parent.mkdir(parents=True, exist_ok=True)

    if not caminho_catalogo.exists():
        print(f"ERRO: catalogo nao encontrado em {caminho_catalogo}. Rode antes o 04_consolidar_curated.py", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(caminho_catalogo)
    print(f"Catalogo carregado: {len(df)} documentos no total")

    if args.apenas_sem_flag and "flag_qualidade" in df.columns:
        antes = len(df)
        df = df[df["flag_qualidade"] == "ok"]
        print(f"Filtrando so documentos com qualidade 'ok': {antes} -> {len(df)}")

    if args.n_amostra > 0 and args.n_amostra < len(df):
        df_amostra = df.sample(n=args.n_amostra, random_state=args.seed).reset_index(drop=True)
        print(f"Amostra aleatoria selecionada: {len(df_amostra)} documentos (seed={args.seed})")
    else:
        df_amostra = df.reset_index(drop=True)
        print(f"Usando o dataset inteiro (sem amostragem): {len(df_amostra)} documentos")

    # --- Carrega texto completo de cada documento da amostra ---
    registros = []
    sem_texto = 0
    for _, row in df_amostra.iterrows():
        id_hash = row["id_hash"]
        texto = carregar_texto(fulltext_dir, id_hash)
        if not texto.strip():
            sem_texto += 1
            continue
        registros.append(
            {
                "id_hash": id_hash,
                "titulo": row.get("titulo", "") or "(sem titulo)",
                "autor": row.get("autor", "") or "(sem autor)",
                "ano": row.get("ano", "") or "(sem ano)",
                "texto": texto,
                "n_paragrafos": contar_paragrafos(texto),
                "n_palavras": contar_palavras(texto),
            }
        )

    if sem_texto:
        print(f"Aviso: {sem_texto} documentos da amostra nao tinham arquivo de texto em {fulltext_dir} e foram ignorados")

    if not registros:
        print("Nenhum documento com texto disponivel. Nada a exportar.")
        sys.exit(0)

    # --- Estatisticas ---
    n_docs = len(registros)
    paragrafos = [r["n_paragrafos"] for r in registros]
    palavras = [r["n_palavras"] for r in registros]

    print("\n--- Estatisticas basicas do dataset (amostra analisada) ---")
    print(f"Numero de documentos:              {n_docs}")
    print(f"Media de paragrafos por documento:  {statistics.mean(paragrafos):.1f}")
    print(f"Mediana de paragrafos por documento: {statistics.median(paragrafos):.1f}")
    print(f"Media de palavras por documento:    {statistics.mean(palavras):.1f}")
    print(f"Mediana de palavras por documento:  {statistics.median(palavras):.1f}")
    print(f"Documento mais curto (palavras):    {min(palavras)}")
    print(f"Documento mais longo (palavras):    {max(palavras)}")
    print(f"Total de palavras no corpus amostrado: {sum(palavras):,}")

    # --- Exporta tudo em um unico .txt com delimitadores ---
    with open(saida_path, "w", encoding="utf-8") as f:
        for r in registros:
            titulo_limpo = r["titulo"].replace('"', "'")[:200]
            f.write(DELIMITADOR_INICIO.format(id_hash=r["id_hash"], titulo=titulo_limpo, ano=r["ano"]) + "\n\n")
            f.write(r["texto"].strip() + "\n\n")
            f.write(DELIMITADOR_FIM.format(id_hash=r["id_hash"]) + "\n\n")

    tamanho_mb = saida_path.stat().st_size / (1024 * 1024)
    print(f"\nArquivo exportado: {saida_path}")
    print(f"Tamanho do arquivo: {tamanho_mb:.1f} MB")
    print(
        "\nFormato de cada documento no .txt:\n"
        "  ===== INICIO_DOCUMENTO id=<hash> titulo=\"...\" ano=<ano> =====\n"
        "  <texto completo do documento>\n"
        "  ===== FIM_DOCUMENTO id=<hash> =====\n"
        "\nPara reabrir e separar por documento depois, faca split pelo padrao"
        " 'INICIO_DOCUMENTO' / 'FIM_DOCUMENTO' com regex."
    )


if __name__ == "__main__":
    main()
