"""
04_consolidar_curated.py

Le todos os JSONs de processed/metadata/, consolida em uma tabela unica
(parquet + csv), aplica:
  - deduplicacao por titulo normalizado (pega duplicatas que NAO sao hash
    identico, ex: mesma tese re-digitalizada ou com metadata de PDF diferente)
  - checagem de qualidade (campos vazios, texto muito curto)
  - flag de revisao manual para quem tem confianca baixa ou parece duplicata

Saida:
  - curated/catalogo.parquet   -> tabela final, uma linha por tese unica
  - curated/catalogo.csv       -> mesma tabela em CSV (mais facil de abrir)
  - curated/possveis_duplicatas.csv -> pares suspeitos de duplicata para revisao manual
  - curated/relatorio_qualidade.txt -> resumo dos problemas encontrados

Dependencias:
    pip install --break-system-packages pandas pyarrow

Uso:
    python 04_consolidar_curated.py \
        --processed-dir teses/processed \
        --curated-dir teses/curated \
        --manifest teses/_control/manifest.csv
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("ERRO: pandas nao instalado. Rode: pip install --break-system-packages pandas pyarrow", file=sys.stderr)
    sys.exit(1)


def normalizar_titulo(titulo: str) -> str:
    """
    Normaliza titulo para comparacao de duplicatas: minusculas, sem acento,
    sem pontuacao, espacos colapsados. Duas teses com o "mesmo" titulo mas
    formatacao diferente vao cair na mesma chave.
    """
    if not titulo:
        return ""
    t = unicodedata.normalize("NFKD", titulo)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def carregar_todos_metadados(meta_dir: Path) -> list:
    registros = []
    arquivos = sorted(meta_dir.glob("*.json"))
    print(f"Carregando {len(arquivos)} arquivos de metadados...")
    for caminho in arquivos:
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                registros.append(json.load(f))
        except Exception as e:
            print(f"  aviso: falha ao ler {caminho.name}: {e}")
    return registros


def detectar_duplicatas_por_titulo(registros: list) -> dict:
    """
    Agrupa registros por titulo normalizado. Grupos com mais de 1 registro
    sao candidatos a duplicata (mesma tese, hashes de arquivo diferentes).
    Retorna dict: titulo_normalizado -> lista de id_hash
    """
    grupos = defaultdict(list)
    for r in registros:
        titulo_norm = normalizar_titulo(r.get("titulo", ""))
        if titulo_norm and len(titulo_norm) > 10:  # ignora titulos vazios/curtos demais
            grupos[titulo_norm].append(r["id_hash"])
    return {k: v for k, v in grupos.items() if len(v) > 1}


def escolher_melhor_do_grupo(registros_grupo: list) -> str:
    """
    Dado um grupo de registros duplicados (mesmo titulo normalizado),
    escolhe qual id_hash manter como 'principal' no catalogo curated:
    prioriza confianca_metadados alta, depois mais paginas, depois mais
    caracteres de texto extraido.
    """
    ordem_confianca = {"alta": 2, "media": 1, "baixa": 0}
    melhor = max(
        registros_grupo,
        key=lambda r: (
            ordem_confianca.get(r.get("confianca_metadados", "baixa"), 0),
            r.get("num_paginas") or 0,
            r.get("num_caracteres_texto") or 0,
        ),
    )
    return melhor["id_hash"]


def main():
    parser = argparse.ArgumentParser(description="Consolida catalogo curated final")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--curated-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--min-caracteres-validos",
        type=int,
        default=500,
        help="Teses com menos texto que isso sao flagadas como 'texto_insuficiente' na qualidade",
    )
    parser.add_argument(
        "--max-tamanho-grupo-auto-dedup",
        type=int,
        default=5,
        help=(
            "Grupos de 'duplicatas' por titulo com mais membros que isso NAO sao "
            "removidos automaticamente - sao so sinalizados para revisao manual. "
            "Protege contra falso positivo (ex: titulo capturado errado = nome de instituicao, "
            "que faz centenas de teses diferentes colapsarem no mesmo grupo)."
        ),
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir).resolve()
    curated_dir = Path(args.curated_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    curated_dir.mkdir(parents=True, exist_ok=True)

    meta_dir = processed_dir / "metadata"
    if not meta_dir.exists():
        print(f"ERRO: {meta_dir} nao encontrado. Rode antes o 03_extrair_metadados.py", file=sys.stderr)
        sys.exit(1)

    registros = carregar_todos_metadados(meta_dir)
    if not registros:
        print("Nenhum registro de metadados encontrado. Nada a fazer.")
        return

    print(f"Total de registros carregados: {len(registros)}")

    # --- Deduplicacao por titulo normalizado ---
    grupos_duplicados = detectar_duplicatas_por_titulo(registros)
    hashes_para_remover = set()
    linhas_duplicatas_relatorio = []

    grupos_suspeitos_grandes = 0
    registros_flagados_sem_remover = set()

    for titulo_norm, id_hashes in grupos_duplicados.items():
        registros_grupo = [r for r in registros if r["id_hash"] in id_hashes]

        # Protecao: grupo grande demais para ser duplicata real de verdade.
        # Mais provavel que seja falso positivo (ex: titulo capturado errado,
        # como nome de instituicao repetido em varias teses diferentes).
        # Nesse caso NAO remove nada, so sinaliza para revisao manual.
        grupo_e_suspeito = len(registros_grupo) > args.max_tamanho_grupo_auto_dedup

        if grupo_e_suspeito:
            grupos_suspeitos_grandes += 1
            for r in registros_grupo:
                linhas_duplicatas_relatorio.append(
                    {
                        "titulo_normalizado": titulo_norm,
                        "id_hash": r["id_hash"],
                        "arquivo_origem": r.get("arquivo_origem", ""),
                        "eh_principal_escolhido": False,
                        "confianca_metadados": r.get("confianca_metadados", ""),
                        "motivo": f"grupo_suspeito_tamanho_{len(registros_grupo)}_nao_removido_automaticamente",
                    }
                )
                registros_flagados_sem_remover.add(r["id_hash"])
            continue

        principal = escolher_melhor_do_grupo(registros_grupo)
        for r in registros_grupo:
            linhas_duplicatas_relatorio.append(
                {
                    "titulo_normalizado": titulo_norm,
                    "id_hash": r["id_hash"],
                    "arquivo_origem": r.get("arquivo_origem", ""),
                    "eh_principal_escolhido": r["id_hash"] == principal,
                    "confianca_metadados": r.get("confianca_metadados", ""),
                    "motivo": "duplicata_removida_automaticamente",
                }
            )
            if r["id_hash"] != principal:
                hashes_para_remover.add(r["id_hash"])

    print(f"Grupos de possiveis duplicatas (por titulo): {len(grupos_duplicados)}")
    print(f"  - Grupos grandes suspeitos (>{args.max_tamanho_grupo_auto_dedup} membros, NAO removidos, so sinalizados): {grupos_suspeitos_grandes}")
    print(f"  - Registros nesses grupos suspeitos (mantidos no catalogo, flag para revisao): {len(registros_flagados_sem_remover)}")
    print(f"Registros que serao removidos do catalogo final (grupos pequenos, mantendo o melhor de cada grupo): {len(hashes_para_remover)}")

    # --- Monta tabela final ---
    registros_finais = [r for r in registros if r["id_hash"] not in hashes_para_remover]

    df = pd.DataFrame(registros_finais)

    # --- Checagem de qualidade ---
    def flag_qualidade(row):
        problemas = []
        if not row.get("titulo"):
            problemas.append("sem_titulo")
        if not row.get("autor"):
            problemas.append("sem_autor")
        if not row.get("ano"):
            problemas.append("sem_ano")
        if (row.get("num_caracteres_texto") or 0) < args.min_caracteres_validos:
            problemas.append("texto_insuficiente")
        if row.get("confianca_metadados") == "baixa":
            problemas.append("confianca_baixa")
        return ";".join(problemas) if problemas else "ok"

    df["flag_qualidade"] = df.apply(flag_qualidade, axis=1)

    # Marca registros de grupos grandes suspeitos tambem, mesmo que a
    # qualidade individual esteja ok - o titulo pode estar errado.
    def adicionar_flag_grupo_suspeito(row):
        base = row["flag_qualidade"]
        if row["id_hash"] in registros_flagados_sem_remover:
            return "titulo_suspeito_grupo_grande" if base == "ok" else base + ";titulo_suspeito_grupo_grande"
        return base

    df["flag_qualidade"] = df.apply(adicionar_flag_grupo_suspeito, axis=1)
    df["precisa_revisao_manual"] = df["flag_qualidade"] != "ok"

    # --- Salva saidas ---
    caminho_parquet = curated_dir / "catalogo.parquet"
    caminho_csv = curated_dir / "catalogo.csv"
    caminho_dup_csv = curated_dir / "possiveis_duplicatas.csv"
    caminho_relatorio = curated_dir / "relatorio_qualidade.txt"

    df.to_parquet(caminho_parquet, index=False)
    df.to_csv(caminho_csv, index=False, encoding="utf-8")

    if linhas_duplicatas_relatorio:
        with open(caminho_dup_csv, "w", encoding="utf-8", newline="") as f:
            escritor = csv.DictWriter(f, fieldnames=list(linhas_duplicatas_relatorio[0].keys()))
            escritor.writeheader()
            escritor.writerows(linhas_duplicatas_relatorio)

    contagem_flags = df["flag_qualidade"].value_counts()
    with open(caminho_relatorio, "w", encoding="utf-8") as f:
        f.write("Relatorio de qualidade - catalogo curated\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total de registros no catalogo final: {len(df)}\n")
        f.write(f"Registros removidos por deduplicacao:  {len(hashes_para_remover)}\n")
        f.write(f"Registros precisando revisao manual:   {int(df['precisa_revisao_manual'].sum())}\n\n")
        f.write("Distribuicao de flags de qualidade:\n")
        for flag, n in contagem_flags.items():
            f.write(f"  {flag}: {n}\n")

    print("\n--- Resumo ---")
    print(f"Catalogo final (parquet):     {caminho_parquet}  ({len(df)} registros)")
    print(f"Catalogo final (csv):         {caminho_csv}")
    if linhas_duplicatas_relatorio:
        print(f"Relatorio de duplicatas:      {caminho_dup_csv}")
    print(f"Relatorio de qualidade:       {caminho_relatorio}")
    print(f"\nRegistros que precisam de revisao manual: {int(df['precisa_revisao_manual'].sum())} de {len(df)}")


if __name__ == "__main__":
    main()