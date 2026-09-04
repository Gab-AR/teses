"""
01_gerar_manifest.py

Varre a pasta raw/pdfs, calcula hash SHA256 de cada PDF, registra metadados
básicos de arquivo e grava tudo em _control/manifest.csv.

Esse manifest é a espinha dorsal do pipeline: cada etapa seguinte (staging,
processed, curated) vai LER esse CSV, atualizar a coluna de status
correspondente, e regravar. Nunca perdemos rastreabilidade de um arquivo.

Uso:
    python 01_gerar_manifest.py --raw-dir teses/raw/pdfs --out teses/_control/manifest.csv
"""

import argparse
import csv
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_arquivo(caminho: Path, bloco: int = 1024 * 1024) -> str:
    """Calcula SHA256 lendo em blocos (não carrega o PDF inteiro na memória)."""
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for chunk in iter(lambda: f.read(bloco), b""):
            h.update(chunk)
    return h.hexdigest()


def carregar_manifest_existente(caminho_csv: Path) -> dict:
    """Se já existe um manifest, carrega para não recalcular hash de tudo de novo."""
    existentes = {}
    if caminho_csv.exists():
        with open(caminho_csv, "r", encoding="utf-8", newline="") as f:
            leitor = csv.DictReader(f)
            for linha in leitor:
                existentes[linha["caminho_relativo"]] = linha
    return existentes


def main():
    parser = argparse.ArgumentParser(description="Gera manifest de PDFs em raw/")
    parser.add_argument("--raw-dir", required=True, help="Pasta com os PDFs originais (raw/pdfs)")
    parser.add_argument("--out", required=True, help="Caminho de saída do manifest.csv")
    parser.add_argument(
        "--forcar-rehash",
        action="store_true",
        help="Recalcula hash mesmo de arquivos já presentes no manifest existente",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_csv = Path(args.out).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"ERRO: pasta raw não encontrada: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Varrendo: {raw_dir}")
    pdfs = sorted(raw_dir.rglob("*.pdf")) + sorted(raw_dir.rglob("*.PDF"))
    pdfs = sorted(set(pdfs))
    total = len(pdfs)
    print(f"Encontrados {total} arquivos .pdf")

    if total == 0:
        print("Nenhum PDF encontrado. Verifique o caminho informado em --raw-dir.")
        sys.exit(0)

    existentes = carregar_manifest_existente(out_csv)
    linhas = []
    hashes_vistos = {}  # hash -> primeiro caminho relativo (pra achar duplicatas)
    duplicatas = 0

    for i, caminho in enumerate(pdfs, start=1):
        rel = str(caminho.relative_to(raw_dir))
        stat = caminho.stat()

        # Reaproveita hash já calculado se o tamanho não mudou e não foi forçado
        anterior = existentes.get(rel)
        if (
            anterior
            and not args.forcar_rehash
            and int(anterior.get("tamanho_bytes", -1)) == stat.st_size
        ):
            hash_sha256 = anterior["sha256"]
        else:
            hash_sha256 = sha256_arquivo(caminho)

        eh_duplicata = hash_sha256 in hashes_vistos
        if eh_duplicata:
            duplicatas += 1
        else:
            hashes_vistos[hash_sha256] = rel

        linhas.append(
            {
                "caminho_relativo": rel,
                "nome_arquivo": caminho.name,
                "tamanho_bytes": stat.st_size,
                "sha256": hash_sha256,
                "duplicata_de": hashes_vistos[hash_sha256] if eh_duplicata else "",
                "data_modificacao": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                "status_extracao_texto": "pendente",
                "status_metadados": "pendente",
                "status_curated": "pendente",
                "data_registro_manifest": datetime.now(timezone.utc).isoformat(),
                "observacoes": "",
            }
        )

        if i % 500 == 0 or i == total:
            print(f"  processados {i}/{total}...")

    campos = [
        "caminho_relativo",
        "nome_arquivo",
        "tamanho_bytes",
        "sha256",
        "duplicata_de",
        "data_modificacao",
        "status_extracao_texto",
        "status_metadados",
        "status_curated",
        "data_registro_manifest",
        "observacoes",
    ]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)

    print("\n--- Resumo ---")
    print(f"Total de PDFs:        {total}")
    print(f"Hashes únicos:        {len(hashes_vistos)}")
    print(f"Duplicatas exatas:    {duplicatas}")
    print(f"Manifest salvo em:    {out_csv}")
    if duplicatas > 0:
        print(
            "\nAtenção: existem duplicatas exatas (mesmo conteúdo/hash)."
            " Elas foram marcadas na coluna 'duplicata_de' e podem ser puladas"
            " nas próximas etapas para não processar a mesma tese duas vezes."
        )


if __name__ == "__main__":
    main()