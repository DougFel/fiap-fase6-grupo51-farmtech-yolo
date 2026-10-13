"""
Monta a página HTML que o vídeo percorre: capa, prints do processo (GitHub, Make
Sense, Colab) e o relatório executado no Colab, sem as células de código.

    python montar_pagina.py

Gera video/pagina_video.html a partir do notebook executado em notebook/.
"""
import base64
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

BASE = Path(__file__).resolve().parent
RAIZ = BASE.parent
NOTEBOOK = RAIZ / "notebook" / "DouglasFelicioDaSilva_rm572312_pbl_fase6.ipynb"
PRINTS = BASE / "prints"


def img64(caminho: Path) -> str:
    tipo = "png" if caminho.suffix == ".png" else "jpeg"
    return f"data:image/{tipo};base64," + base64.b64encode(caminho.read_bytes()).decode()


def secao_print(titulo: str, arquivo: str, legenda: str) -> str:
    return (f'<section class="print"><h2>{titulo}</h2>'
            f'<img src="{img64(PRINTS / arquivo)}" alt="{titulo}"><p>{legenda}</p></section>')


def main() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    exp = HTMLExporter(template_name="lab")
    exp.exclude_input = True                     # só textos e saídas: o código está no notebook
    exp.exclude_input_prompt = True
    exp.exclude_output_prompt = True
    corpo, _ = exp.from_notebook_node(nb)

    capa = """
<section class="capa">
  <div class="fase">FIAP · Inteligência Artificial · Fase 6 · Cap 1</div>
  <h1>FarmTech Solutions — Visão Computacional com YOLO</h1>
  <p class="sub">Gado × maçã: YOLOv5 customizado, YOLO tradicional e CNN do zero</p>
  <div class="grupo"><strong>Grupo 51</strong><br>
  Douglas Felicio da Silva — RM572312<br>
  Gabriel Kaique Portel Silva — RM572500<br>
  Richard Wrobel dos Santos — RM573998</div>
</section>"""
    processo = (
        secao_print("O repositório no GitHub", "github.png",
                    "Repositório público do grupo: notebook, base de imagens, rótulos e scripts.")
        + secao_print("Rotulagem no Make Sense", "makesense.png",
                      "80 imagens rotuladas (✓), uma caixa por objeto, exportadas em formato YOLO.")
        + secao_print("Execução no Google Colab", "colab.jpg",
                      "Notebook conectado ao Google Drive, executado com GPU Tesla T4.")
    )
    estilo = """
<style>
  body { background: #ffffff; }
  .capa { min-height: 860px; display: flex; flex-direction: column; justify-content: center;
          padding: 0 120px; border-left: 14px solid #ed145b; }
  .capa .fase { color: #ed145b; font: 700 20px/1.4 -apple-system, sans-serif; letter-spacing: .14em;
                text-transform: uppercase; }
  .capa h1 { font: 700 54px/1.15 -apple-system, sans-serif; margin: 18px 0 10px; color: #1a1d21; }
  .capa .sub { font: 400 26px/1.4 -apple-system, sans-serif; color: #5c6570; margin: 0 0 40px; }
  .capa .grupo { font: 400 24px/1.6 -apple-system, sans-serif; color: #1a1d21;
                 background: #f5f6f8; padding: 22px 28px; border-radius: 8px; width: fit-content; }
  .print { padding: 40px 80px 60px; text-align: center; }
  .print h2 { font: 700 34px/1.2 -apple-system, sans-serif; color: #1a1d21; text-align: left; }
  .print img { max-width: 100%; max-height: 640px; border: 1px solid #dde1e6; border-radius: 8px;
               box-shadow: 0 6px 24px rgba(0,0,0,.08); }
  .print p { font: 400 20px/1.4 -apple-system, sans-serif; color: #5c6570; }
  .jp-Notebook { max-width: 1400px; margin: 0 auto; }
  .jp-RenderedHTMLCommon { font-size: 17px !important; }
  .jp-RenderedImage img { max-width: 100% !important; }
</style>"""
    html = corpo.replace("<body", "<body", 1)
    html = html.replace("</head>", estilo + "</head>", 1)
    marcador = '<main>' if '<main>' in html else '<body class="jp-Notebook"'
    i = html.find('>', html.find(marcador)) + 1
    html = html[:i] + capa + processo + html[i:]
    destino = BASE / "pagina_video.html"
    destino.write_text(html, encoding="utf-8")
    print(f"{destino.name}: {destino.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
