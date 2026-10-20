#!/usr/bin/env python3
"""
download_mawi.py - Descarga automatizada de las capturas crudas PCAP de MAWI (Punto F, 2021).
Uso: python src/download_mawi.py [--all] [--date YYYYMMDD]
"""

import os
import sys
import argparse
import urllib.request
from typing import List

MAWI_BASE_URL = "http://mawi.wide.ad.jp/mawi/samplepoint-F/2021"

CAPTURES = {
    "20210210": {
        "file": "202102101400.pcap.gz",
        "size_gb": 2.65,
        "url": f"{MAWI_BASE_URL}/202102101400.pcap.gz"
    },
    "20210616": {
        "file": "202106161400.pcap.gz",
        "size_gb": 1.69,
        "url": f"{MAWI_BASE_URL}/202106161400.pcap.gz"
    },
    "20211013": {
        "file": "202110131400.pcap.gz",
        "size_gb": 1.81,
        "url": f"{MAWI_BASE_URL}/202110131400.pcap.gz"
    }
}

def download_file(url: str, output_path: str):
    if os.path.exists(output_path):
        print(f"[OK] El archivo ya existe localmente: {output_path} ({os.path.getsize(output_path) / 1e9:.2f} GB)")
        return

    print(f"[DESCARGANDO] {url} -> {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 100 / totalsize
            s = f"\rProgreso: {percent:5.1f}% ({readsofar / 1e6:.1f} MB de {totalsize / 1e6:.1f} MB)"
            sys.stderr.write(s)
            if readsofar >= totalsize:
                sys.stderr.write("\n")
        else:
            sys.stderr.write(f"\rLeídos: {readsofar / 1e6:.1f} MB")

    urllib.request.urlretrieve(url, output_path, reporthook)
    print(f"\n[COMPLETADO] Guardado exitosamente en: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Descargador de trazas crudas PCAP de MAWI WIDE 2021.")
    parser.add_argument("--all", action="store_true", help="Descargar las 3 capturas del estudio (6.15 GB total)")
    parser.add_argument("--date", type=str, choices=list(CAPTURES.keys()), help="Fecha específica a descargar (YYYYMMDD)")
    parser.add_argument("--dest", type=str, default="data/raw/pcap", help="Directorio destino para guardar los archivos PCAP")

    args = parser.parse_args()

    targets: List[str] = []
    if args.date:
        targets.append(args.date)
    elif args.all:
        targets = list(CAPTURES.keys())
    else:
        print("Especifique --all para descargar las 3 capturas o --date YYYYMMDD para una en particular.")
        print("Fechas disponibles:", list(CAPTURES.keys()))
        sys.exit(1)

    print(f"Iniciando descarga en: {os.path.abspath(args.dest)}")
    for date in targets:
        info = CAPTURES[date]
        dest_file = os.path.join(args.dest, info["file"])
        print(f"\n--- Captura: {date} ({info['size_gb']} GB) ---")
        try:
            download_file(info["url"], dest_file)
        except Exception as e:
            print(f"[ERROR] No se pudo descargar {info['file']}: {e}")

if __name__ == "__main__":
    main()
