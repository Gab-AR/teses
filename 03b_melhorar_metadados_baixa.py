"""
03b_melhorar_metadados_baixa.py

Reprocessa APENAS os registros com status_metadados == 'ok_baixa' no manifest,
usando heuristicas mais fortes que o 03_extrair_metadados.py original:

  1) Titulo por TAMANHO DE FONTE: usa PyMuPDF para achar o texto com a maior
     fonte nas primeiras paginas (capa) - muito mais confiavel que "primeira
     linha longa" quando o layout da capa e mais elaborado.
  2) Busca em mais paginas (nao so a primeira) para autor/orientador/
     instituicao/ano, cobrindo capa + folha de rosto + ficha catalografica.
  3. Regex mais especifica para o padrao de teses/dissertacoes brasileiras
     (ex: "Programa de Pos-Graduacao em", "para obtencao do titulo de
     Doutor/Mestre em").

So SOBRESCREVE o JSON/status se a nova extracao tiver confianca igual ou
maior que a anterior. Nunca piora um registro.

Dependencias:
    pip install --break-system-packages pymupdf

Uso:
    python 03b_melhorar_metadados_baixa.py \
        --raw-dir teses/raw/pdfs \
        --manifest teses/_control/manifest.csv \
        --processed-dir teses/processed \
        --paginas-analisadas 4
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

try:
    import fitz
except ImportError:
    print("ERRO: PyMuPDF nao instalado. Rode: pip install --break-system-packages pymupdf", file=sys.stderr)
    sys.exit(1)


RE_ANO = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")

RE_AUTOR = re.compile(
    r"(?:^|\n)\s*(?:por|autor(?:a)?)\s*[:\-]?\s*([A-ZÀ-Ú][^\n]{4,120})", re.IGNORECASE
)

RE_ORIENTADOR = re.compile(
    r"orientador(?:a)?\s*[:\-]?\s*([A-ZÀ-Ú][^\n]{4,120})", re.IGNORECASE
)

RE_INSTITUICAO = re.compile(
    r"(universidade[^\n]{0,90}|instituto\s+(?:federal|superior)[^\n]{0,90}|faculdade[^\n]{0,90}|centro\s+universitario[^\n]{0,90})",
    re.IGNORECASE,
)

# Padroes tipicos de teses/dissertacoes brasileiras - ajudam a achar o bloco
# certo da pagina (perto de "Programa de Pos-Graduacao", "obtencao do titulo")
RE_CONTEXTO_PROGRAMA = re.compile(
    r"programa\s+de\s+p[oó]s[- ]?gradua[cç][aã]o\s+em\s+([^\n]{3,100})",
    re.IGNORECASE,
)
RE_CONTEXTO_TITULO_GRAU = re.compile(
    r"(?:para\s+)?obten[cç][aã]o\s+do\s+t[ií]tulo\s+de\s+(doutor|mestre)[^\n]{0,80}",
    re.IGNORECASE,
)


def sem_acento(txt: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", txt) if not unicodedata.combining(c))


def parece_nome_de_instituicao_ou_ruido(texto_linha: str) -> bool:
    """
    Filtra linhas que NAO devem virar titulo mesmo estando em fonte grande:
    nome de universidade/instituto/faculdade, orgaos publicos, brasoes com
    texto (ex: "MINISTERIO DA EDUCACAO"), ou linhas so em maiusculas curtas
    (sigla/logo). Capas de tese quase sempre tem o nome da instituicao em
    destaque no topo, MAIOR que o titulo em muitos casos - por isso essa
    checagem e essencial.
    """
    t = sem_acento(texto_linha).lower()

    padroes_instituicao = (
        "universidade", "instituto federal", "instituto superior",
        "faculdade", "centro universitario", "ministerio da educacao",
        "ministerio", "secretaria de", "campus", "pro-reitoria",
        "coordenacao de", "programa de pos-graduacao", "pos-graduacao em",
        "pro reitoria",
    )
    if any(p in t for p in padroes_instituicao):
        return True

    # Linha inteira em maiusculas e curta -> provavel sigla/logo, nao titulo
    if texto_linha.isupper() and len(texto_linha) < 25:
        return True

    return False


def extrair_titulo_por_fonte(caminho_pdf: Path, n_paginas: int = 2) -> str:
    """
    Percorre as primeiras n_paginas e acha o bloco de texto com a MAIOR fonte
    QUE NAO seja nome de instituicao/ruido (ver parece_nome_de_instituicao_ou_ruido).
    Em capas de tese, o titulo costuma ser o maior texto relevante, mas o nome
    da universidade no topo as vezes esta em fonte igual ou maior - por isso
    filtramos candidatos, nao so pegamos o maior de todos.
    Ignora blocos muito curtos (provavel logo/sigla) ou muito longos (provavel
    paragrafo de resumo em fonte grande por engano de deteccao).
    """
    candidatos = []  # (tamanho_fonte, texto)

    try:
        with fitz.open(caminho_pdf) as doc:
            for i, pagina in enumerate(doc):
                if i >= n_paginas:
                    break
                dados = pagina.get_text("dict")
                for bloco in dados.get("blocks", []):
                    for linha in bloco.get("lines", []):
                        texto_linha = "".join(s["text"] for s in linha.get("spans", []))
                        texto_linha = texto_linha.strip()
                        if not texto_linha or len(texto_linha) < 15 or len(texto_linha) > 250:
                            continue
                        if parece_nome_de_instituicao_ou_ruido(texto_linha):
                            continue
                        tamanho = max((s.get("size", 0) for s in linha.get("spans", [])), default=0)
                        candidatos.append((tamanho, texto_linha))
    except Exception:
        return ""

    if not candidatos:
        return ""

    candidatos.sort(key=lambda x: x[0], reverse=True)
    return candidatos[0][1]


def extrair_texto_primeiras_paginas(caminho_pdf: Path, n_paginas: int) -> str:
    try:
        with fitz.open(caminho_pdf) as doc:
            partes = []
            for i, pagina in enumerate(doc):
                if i >= n_paginas:
                    break
                partes.append(pagina.get_text())
            return "\n".join(partes)
    except Exception:
        return ""


def extrair_metadados_melhorados(caminho_pdf: Path, n_paginas: int) -> dict:
    texto_paginas = extrair_texto_primeiras_paginas(caminho_pdf, n_paginas)
    titulo_fonte = extrair_titulo_por_fonte(caminho_pdf, n_paginas=min(2, n_paginas))

    resultado = {
        "titulo": titulo_fonte,
        "autor": "",
        "orientador": "",
        "instituicao": "",
        "ano": "",
    }

    m_autor = RE_AUTOR.search(texto_paginas)
    if m_autor:
        resultado["autor"] = m_autor.group(1).strip()[:150]

    m_orient = RE_ORIENTADOR.search(texto_paginas)
    if m_orient:
        resultado["orientador"] = m_orient.group(1).strip()[:150]

    m_inst = RE_INSTITUICAO.search(texto_paginas)
    if m_inst:
        resultado["instituicao"] = m_inst.group(1).strip()[:200]
    else:
        m_programa = RE_CONTEXTO_PROGRAMA.search(texto_paginas)
        if m_programa:
            resultado["instituicao"] = f"Programa de Pos-Graduacao em {m_programa.group(1).strip()[:150]}"

    m_ano = RE_ANO.search(texto_paginas)
    if m_ano:
        resultado["ano"] = m_ano.group(1)

    resultado["achou_contexto_tese"] = bool(RE_CONTEXTO_TITULO_GRAU.search(texto_paginas))

    return resultado


def calcular_confianca(reg: dict) -> str:
    campos_preenchidos = sum(bool(reg.get(c)) for c in ("titulo", "autor", "ano"))
    if campos_preenchidos == 3 and reg.get("instituicao"):
        return "alta"
    elif campos_preenchidos >= 2:
        return "media"
    else:
        return "baixa"


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
    parser = argparse.ArgumentParser(description="Reprocessa registros ok_baixa com heuristicas melhoradas")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--paginas-analisadas", type=int, default=4, help="Quantas primeiras paginas olhar (capa + folha de rosto + ficha catalografica)")
    parser.add_argument("--limite", type=int, default=0)
    parser.add_argument("--salvar-a-cada", type=int, default=100)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    processed_dir = Path(args.processed_dir).resolve()
    meta_dir = processed_dir / "metadata"

    campos, linhas = ler_manifest(manifest_path)
    alvo = [l for l in linhas if l.get("status_metadados") == "ok_baixa"]
    print(f"Registros ok_baixa encontrados: {len(alvo)}")

    if args.limite > 0:
        alvo = alvo[: args.limite]
        print(f"Limitando a {len(alvo)} (--limite)")

    if not alvo:
        print("Nada para reprocessar.")
        return

    melhorados = 0
    mantidos = 0

    for i, linha in enumerate(alvo, start=1):
        hash_sha256 = linha["sha256"]
        rel = linha["caminho_relativo"]
        caminho_pdf = raw_dir / rel
        caminho_json = meta_dir / f"{hash_sha256}.json"

        if not caminho_pdf.exists() or not caminho_json.exists():
            continue

        registro_antigo = json.loads(caminho_json.read_text(encoding="utf-8"))

        novos_campos = extrair_metadados_melhorados(caminho_pdf, args.paginas_analisadas)
        nova_confianca = calcular_confianca(novos_campos)

        ordem = {"alta": 2, "media": 1, "baixa": 0}
        if ordem[nova_confianca] > ordem.get(registro_antigo.get("confianca_metadados", "baixa"), 0):
            # Atualiza so os campos de metadados, preserva o resto (texto, contagens etc)
            registro_antigo["titulo"] = novos_campos["titulo"] or registro_antigo.get("titulo", "")
            registro_antigo["autor"] = novos_campos["autor"] or registro_antigo.get("autor", "")
            registro_antigo["orientador"] = novos_campos["orientador"] or registro_antigo.get("orientador", "")
            registro_antigo["instituicao"] = novos_campos["instituicao"] or registro_antigo.get("instituicao", "")
            registro_antigo["ano"] = novos_campos["ano"] or registro_antigo.get("ano", "")
            registro_antigo["confianca_metadados"] = nova_confianca
            registro_antigo["metodo_extracao"] = "heuristica_v2_fonte_multipagina"

            caminho_json.write_text(json.dumps(registro_antigo, ensure_ascii=False, indent=2), encoding="utf-8")
            linha["status_metadados"] = f"ok_{nova_confianca}"
            melhorados += 1
        else:
            mantidos += 1

        if i % args.salvar_a_cada == 0 or i == len(alvo):
            salvar_manifest(manifest_path, campos, linhas)
            print(f"  [{i}/{len(alvo)}] melhorados={melhorados} mantidos_como_baixa={mantidos}")

    salvar_manifest(manifest_path, campos, linhas)

    print("\n--- Resumo ---")
    print(f"Melhorados (confianca subiu):  {melhorados}")
    print(f"Continuam ok_baixa:             {mantidos}")
    print(
        "\nOs que continuam 'baixa' provavelmente sao PDFs com capa fora do"
        " padrao (ex: teses estrangeiras, capas so-imagem sem texto, ou"
        " OCR de baixa qualidade). Esses valem revisao manual ou uma amostra"
        " para decidir se compensa mais uma rodada de regras."
    )


if __name__ == "__main__":
    main()