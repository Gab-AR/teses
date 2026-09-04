"""
03_extrair_metadados.py

Le o manifest.csv + os textos em staging/text_extracted/, tenta extrair
metadados (titulo, autor, instituicao, ano, orientador) usando:
  1) metadados embutidos no PDF (via PyMuPDF)
  2) heuristicas de regex na primeira(s) pagina(s) do texto
  3) fallback no nome do arquivo

Grava um JSON por tese em processed/metadata/<hash>.json e o texto "limpo"
(normalizado) em processed/fulltext/<hash>.txt.

Cada JSON registra 'confianca_metadados' (alta/media/baixa) para voce saber
quais teses precisam de revisao manual depois, sem ter que checar as 20 mil.

Dependencias:
    pip install --break-system-packages pymupdf

Uso:
    python 03_extrair_metadados.py \
        --raw-dir teses/raw/pdfs \
        --manifest teses/_control/manifest.csv \
        --staging-dir teses/staging \
        --processed-dir teses/processed \
        --limite 0
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERRO: PyMuPDF nao instalado. Rode: pip install --break-system-packages pymupdf", file=sys.stderr)
    sys.exit(1)


# --- Heuristicas de regex para a primeira pagina ------------------------

RE_ANO = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
RE_AUTOR = re.compile(
    r"(?:^|\n)\s*(?:por|autor(?:a)?)\s*[:\-]?\s*(.+)", re.IGNORECASE
)
RE_ORIENTADOR = re.compile(
    r"orientador(?:a)?\s*[:\-]?\s*(.+)", re.IGNORECASE
)
RE_INSTITUICAO = re.compile(
    r"(universidade[^\n]{0,80}|instituto[^\n]{0,80}|faculdade[^\n]{0,80})",
    re.IGNORECASE,
)

PALAVRAS_TITULO_INVALIDAS = {"tese", "dissertacao", "dissertação", "monografia"}

# Linhas contendo qualquer um desses termos nunca sao aceitas como titulo,
# mesmo sendo "longas" - sao nome de instituicao/orgao, nao titulo da tese.
# Sem esse filtro, centenas de teses diferentes acabam com o mesmo "titulo"
# (o nome da universidade) e sao erroneamente agrupadas como duplicatas.
PADROES_INSTITUICAO_EM_TITULO = (
    "universidade", "instituto federal", "instituto superior", "faculdade",
    "centro universitario", "centro universitário", "ministerio da educacao",
    "ministério da educação", "secretaria de", "campus", "pro-reitoria",
    "pró-reitoria", "coordenacao de", "coordenação de",
    "programa de pos-graduacao", "programa de pós-graduação",
)


def normalizar_texto(txt: str) -> str:
    """Remove espacos redundantes e caracteres de controle, mantendo acentos."""
    txt = txt.replace("\x00", " ")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def sem_acento(txt: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", txt) if not unicodedata.combining(c)
    )


def extrair_metadados_pdf(caminho_pdf: Path) -> dict:
    """Metadados embutidos no arquivo PDF (nem sempre preenchidos)."""
    try:
        with fitz.open(caminho_pdf) as doc:
            meta = doc.metadata or {}
            return {
                "titulo_pdf": (meta.get("title") or "").strip(),
                "autor_pdf": (meta.get("author") or "").strip(),
                "data_criacao_pdf": (meta.get("creationDate") or "").strip(),
                "num_paginas": doc.page_count,
            }
    except Exception:
        return {"titulo_pdf": "", "autor_pdf": "", "data_criacao_pdf": "", "num_paginas": None}


def extrair_metadados_texto(primeira_pagina: str) -> dict:
    """Heuristicas de regex na primeira pagina do texto extraido."""
    resultado = {"titulo_regex": "", "autor_regex": "", "orientador_regex": "", "instituicao_regex": "", "ano_regex": ""}

    m_ano = RE_ANO.search(primeira_pagina)
    if m_ano:
        resultado["ano_regex"] = m_ano.group(1)

    m_autor = RE_AUTOR.search(primeira_pagina)
    if m_autor:
        resultado["autor_regex"] = m_autor.group(1).strip()[:150]

    m_orient = RE_ORIENTADOR.search(primeira_pagina)
    if m_orient:
        resultado["orientador_regex"] = m_orient.group(1).strip()[:150]

    m_inst = RE_INSTITUICAO.search(primeira_pagina)
    if m_inst:
        resultado["instituicao_regex"] = m_inst.group(1).strip()[:200]

    # Titulo: heuristica simples - primeira linha "longa" (>15 chars) que nao
    # seja so a palavra "TESE"/"DISSERTACAO" e nao seja uma linha em maiusculas
    # curta (geralmente nome de universidade no topo).
    linhas = [l.strip() for l in primeira_pagina.split("\n") if l.strip()]
    for linha in linhas:
        linha_normalizada = sem_acento(linha).lower()
        if len(linha) < 15:
            continue
        if any(p in linha_normalizada for p in PALAVRAS_TITULO_INVALIDAS) and len(linha) < 40:
            continue
        if any(p in linha_normalizada for p in PADROES_INSTITUICAO_EM_TITULO):
            continue
        if linha.isupper() and len(linha) < 25:
            continue
        resultado["titulo_regex"] = linha[:300]
        break

    return resultado


def extrair_do_nome_arquivo(nome_arquivo: str) -> dict:
    """Ultimo fallback: tenta achar um ano no proprio nome do arquivo."""
    m_ano = RE_ANO.search(nome_arquivo)
    return {"ano_nome_arquivo": m_ano.group(1) if m_ano else ""}


def montar_registro(hash_sha256, rel, texto_completo, meta_pdf, meta_regex, meta_nome, num_chars_ocr_flag):
    primeira_pagina = texto_completo[:3000]

    titulo = meta_pdf["titulo_pdf"] or meta_regex["titulo_regex"] or ""
    autor = meta_pdf["autor_pdf"] or meta_regex["autor_regex"] or ""
    ano = meta_regex["ano_regex"] or meta_nome["ano_nome_arquivo"] or ""

    # Confianca: alta se veio de metadados nativos do PDF (mais confiavel),
    # media se veio de regex no texto, baixa se sobrou so o nome do arquivo
    # ou campos ficaram vazios.
    if meta_pdf["titulo_pdf"] and meta_pdf["autor_pdf"]:
        confianca = "alta"
    elif titulo and autor:
        confianca = "media"
    else:
        confianca = "baixa"

    return {
        "id_hash": hash_sha256,
        "arquivo_origem": rel,
        "titulo": titulo,
        "autor": autor,
        "orientador": meta_regex["orientador_regex"],
        "instituicao": meta_regex["instituicao_regex"],
        "ano": ano,
        "num_paginas": meta_pdf["num_paginas"],
        "num_caracteres_texto": len(texto_completo.strip()),
        "confianca_metadados": confianca,
        "trecho_primeira_pagina": primeira_pagina[:500],
    }


def ler_manifest(caminho_csv: Path):
    with open(caminho_csv, "r", encoding="utf-8", newline="") as f:
        leitor = csv.DictReader(f)
        campos = leitor.fieldnames
        linhas = list(leitor)
    return campos, linhas


def salvar_manifest(caminho_csv: Path, campos, linhas):
    tmp = caminho_csv.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)
    tmp.replace(caminho_csv)


def main():
    parser = argparse.ArgumentParser(description="Extrai metadados estruturados (processed)")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--staging-dir", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--limite", type=int, default=0)
    parser.add_argument("--salvar-a-cada", type=int, default=100)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    staging_dir = Path(args.staging_dir).resolve()
    processed_dir = Path(args.processed_dir).resolve()

    texto_dir = staging_dir / "text_extracted"
    meta_out_dir = processed_dir / "metadata"
    fulltext_out_dir = processed_dir / "fulltext"
    meta_out_dir.mkdir(parents=True, exist_ok=True)
    fulltext_out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"ERRO: manifest nao encontrado em {manifest_path}", file=sys.stderr)
        sys.exit(1)

    campos, linhas = ler_manifest(manifest_path)
    if "status_metadados" not in campos:
        campos.append("status_metadados")

    elegiveis = [
        l for l in linhas
        if l["status_extracao_texto"] in ("ok", "ok_ocr")
        and l.get("status_metadados", "pendente") == "pendente"
        and not l.get("duplicata_de")
    ]
    print(f"Elegiveis para extracao de metadados: {len(elegiveis)}")

    if args.limite > 0:
        elegiveis = elegiveis[: args.limite]
        print(f"Limitando a {len(elegiveis)} (--limite)")

    if not elegiveis:
        print("Nada a processar. Rode antes o 02_extrair_texto.py se ainda nao rodou.")
        return

    processados = 0
    sem_texto_staging = 0

    for i, linha in enumerate(elegiveis, start=1):
        rel = linha["caminho_relativo"]
        hash_sha256 = linha["sha256"]
        caminho_txt = texto_dir / f"{hash_sha256}.txt"
        caminho_pdf = raw_dir / rel

        if not caminho_txt.exists():
            linha["status_metadados"] = "falhou_sem_texto_staging"
            sem_texto_staging += 1
            continue

        texto_bruto = caminho_txt.read_text(encoding="utf-8", errors="ignore")
        texto_limpo = normalizar_texto(texto_bruto)

        meta_pdf = extrair_metadados_pdf(caminho_pdf) if caminho_pdf.exists() else {
            "titulo_pdf": "", "autor_pdf": "", "data_criacao_pdf": "", "num_paginas": None
        }
        meta_regex = extrair_metadados_texto(texto_limpo[:3000])
        meta_nome = extrair_do_nome_arquivo(linha["nome_arquivo"])

        registro = montar_registro(hash_sha256, rel, texto_limpo, meta_pdf, meta_regex, meta_nome, None)

        # Grava JSON de metadados
        (meta_out_dir / f"{hash_sha256}.json").write_text(
            json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Grava texto limpo na camada processed (separado do staging bruto)
        (fulltext_out_dir / f"{hash_sha256}.txt").write_text(texto_limpo, encoding="utf-8")

        linha["status_metadados"] = f"ok_{registro['confianca_metadados']}"
        processados += 1

        if i % args.salvar_a_cada == 0 or i == len(elegiveis):
            salvar_manifest(manifest_path, campos, linhas)
            print(f"  [{i}/{len(elegiveis)}] processados={processados} sem_texto_staging={sem_texto_staging}")

    salvar_manifest(manifest_path, campos, linhas)

    print("\n--- Resumo ---")
    print(f"Metadados extraidos:      {processados}")
    print(f"Sem texto no staging:     {sem_texto_staging} (rode 02_extrair_texto.py para esses)")
    print(f"JSONs salvos em:          {meta_out_dir}")
    print(f"Textos limpos salvos em:  {fulltext_out_dir}")
    print(
        "\nDica: confira quantos registros ficaram com status 'ok_baixa' no manifest"
        " (coluna status_metadados) - esses sao os que mais precisam de revisao manual"
        " ou de um metodo extra de extracao (ex: cruzar com API da BDTD/OpenAlex por titulo)."
    )


if __name__ == "__main__":
    main()