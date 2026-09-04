"""
02_extrair_texto.py

Le o manifest.csv, percorre os PDFs com status 'pendente' em
status_extracao_texto, tenta extrair texto com PyMuPDF (rapido) e, se o
resultado vier muito curto (sinal de PDF escaneado/imagem), cai para OCR
via pytesseract + pdf2image.

Saida:
  - Texto bruto extraido vai para staging/text_extracted/<hash>.txt
  - Log de erros vai para staging/logs/erros_extracao.log
  - manifest.csv e atualizado (status_extracao_texto: ok / ok_ocr / falhou)

Dependencias:
    pip install --break-system-packages pymupdf pdf2image pytesseract
    # OCR tambem exige os binarios do sistema:
    #   Ubuntu/Debian: sudo apt install tesseract-ocr poppler-utils
    #   Mac:           brew install tesseract poppler
    #   Windows:       instalar Tesseract-OCR e Poppler manualmente e ajustar PATH

Uso:
    python 02_extrair_texto.py \
        --raw-dir teses/raw/pdfs \
        --manifest teses/_control/manifest.csv \
        --staging-dir teses/staging \
        --min-chars-sem-ocr 200 \
        --limite 0
"""

import argparse
import csv
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERRO: PyMuPDF nao instalado. Rode: pip install --break-system-packages pymupdf", file=sys.stderr)
    sys.exit(1)


def extrair_com_pymupdf(caminho_pdf: Path) -> str:
    """Extrai texto corrido de todas as paginas usando PyMuPDF."""
    texto_paginas = []
    with fitz.open(caminho_pdf) as doc:
        for pagina in doc:
            texto_paginas.append(pagina.get_text())
    return "\n\n".join(texto_paginas)


def extrair_com_ocr(caminho_pdf: Path, dpi: int = 200) -> str:
    """
    Fallback para PDFs escaneados: renderiza cada pagina como imagem e roda
    OCR. Mais lento, so deve ser chamado quando o PyMuPDF falha ou retorna
    pouco texto.
    """
    from pdf2image import convert_from_path
    import pytesseract

    texto_paginas = []
    imagens = convert_from_path(str(caminho_pdf), dpi=dpi)
    for img in imagens:
        texto_paginas.append(pytesseract.image_to_string(img, lang="por"))
    return "\n\n".join(texto_paginas)


def ler_manifest(caminho_csv: Path):
    with open(caminho_csv, "r", encoding="utf-8", newline="") as f:
        leitor = csv.DictReader(f)
        campos = leitor.fieldnames
        linhas = list(leitor)
    return campos, linhas


def salvar_manifest(caminho_csv: Path, campos, linhas):
    # Salva em arquivo temporario e substitui, para nao corromper o manifest
    # se o processo for interrompido no meio da escrita.
    tmp = caminho_csv.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)
    tmp.replace(caminho_csv)


def main():
    parser = argparse.ArgumentParser(description="Extrai texto dos PDFs (staging)")
    parser.add_argument("--raw-dir", required=True, help="Pasta raw/pdfs")
    parser.add_argument("--manifest", required=True, help="Caminho do manifest.csv")
    parser.add_argument("--staging-dir", required=True, help="Pasta staging (ex: teses/staging)")
    parser.add_argument(
        "--min-chars-sem-ocr",
        type=int,
        default=200,
        help="Se o texto extraido pelo PyMuPDF tiver menos caracteres que isso, tenta OCR",
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=0,
        help="Processa no maximo N arquivos nesta execucao (0 = sem limite). Util para testar antes de rodar tudo.",
    )
    parser.add_argument(
        "--pular-ocr",
        action="store_true",
        help="Nao tenta OCR, so marca como 'falhou' quando o PyMuPDF nao extrai texto suficiente",
    )
    parser.add_argument(
        "--salvar-a-cada",
        type=int,
        default=50,
        help="Regrava o manifest no disco a cada N arquivos processados (checkpoint)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    staging_dir = Path(args.staging_dir).resolve()
    texto_dir = staging_dir / "text_extracted"
    logs_dir = staging_dir / "logs"
    texto_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "erros_extracao.log"

    if not manifest_path.exists():
        print(f"ERRO: manifest nao encontrado em {manifest_path}. Rode antes o 01_gerar_manifest.py", file=sys.stderr)
        sys.exit(1)

    campos, linhas = ler_manifest(manifest_path)

    pendentes = [l for l in linhas if l["status_extracao_texto"] == "pendente" and not l.get("duplicata_de")]
    print(f"Total no manifest: {len(linhas)}")
    print(f"Pendentes (excluindo duplicatas exatas): {len(pendentes)}")

    if args.limite > 0:
        pendentes = pendentes[: args.limite]
        print(f"Limitando esta execucao a {len(pendentes)} arquivos (--limite)")

    if not pendentes:
        print("Nada a processar. Todos os arquivos ja foram tentados.")
        return

    ok = ok_ocr = falhou = 0

    with open(log_path, "a", encoding="utf-8") as log_f:
        for i, linha in enumerate(pendentes, start=1):
            rel = linha["caminho_relativo"]
            caminho_pdf = raw_dir / rel
            hash_sha256 = linha["sha256"]
            saida_txt = texto_dir / f"{hash_sha256}.txt"

            try:
                if not caminho_pdf.exists():
                    raise FileNotFoundError(f"Arquivo listado no manifest nao existe em disco: {caminho_pdf}")

                texto = extrair_com_pymupdf(caminho_pdf)

                if len(texto.strip()) < args.min_chars_sem_ocr and not args.pular_ocr:
                    # Provavel PDF escaneado / imagem -> tenta OCR
                    texto_ocr = extrair_com_ocr(caminho_pdf)
                    if len(texto_ocr.strip()) > len(texto.strip()):
                        texto = texto_ocr
                        linha["status_extracao_texto"] = "ok_ocr"
                        ok_ocr += 1
                    else:
                        linha["status_extracao_texto"] = "ok" if texto.strip() else "falhou"
                        ok += 1 if texto.strip() else 0
                        falhou += 0 if texto.strip() else 1
                else:
                    linha["status_extracao_texto"] = "ok"
                    ok += 1

                saida_txt.write_text(texto, encoding="utf-8")
                linha["observacoes"] = f"chars_extraidos={len(texto.strip())}"

            except Exception as e:
                linha["status_extracao_texto"] = "falhou"
                linha["observacoes"] = f"erro: {type(e).__name__}: {e}"
                falhou += 1
                log_f.write(
                    f"[{datetime.now(timezone.utc).isoformat()}] {rel}\n"
                    f"{traceback.format_exc()}\n{'-'*80}\n"
                )

            if i % args.salvar_a_cada == 0 or i == len(pendentes):
                salvar_manifest(manifest_path, campos, linhas)
                print(f"  [{i}/{len(pendentes)}] ok={ok} ok_ocr={ok_ocr} falhou={falhou} (checkpoint salvo)")

    salvar_manifest(manifest_path, campos, linhas)

    print("\n--- Resumo desta execucao ---")
    print(f"Extraidos direto (PyMuPDF):  {ok}")
    print(f"Extraidos via OCR:            {ok_ocr}")
    print(f"Falharam:                     {falhou}")
    print(f"Textos salvos em:             {texto_dir}")
    print(f"Log de erros em:              {log_path}")
    print(f"Manifest atualizado em:       {manifest_path}")


if __name__ == "__main__":
    main()