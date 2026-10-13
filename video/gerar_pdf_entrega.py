"""Gera o PDF de entrega do portal, com os integrantes e os links do repositório e do vídeo."""
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = Path(__file__).resolve().parent.parent
REPO = "https://github.com/DougFel/fiap-fase6-grupo51-farmtech-yolo"
NOTEBOOK = REPO + "/blob/main/notebook/DouglasFelicioDaSilva_rm572312_pbl_fase6.ipynb"
COLAB = ("https://colab.research.google.com/github/DougFel/fiap-fase6-grupo51-farmtech-yolo/"
         "blob/main/notebook/DouglasFelicioDaSilva_rm572312_pbl_fase6.ipynb")
VIDEO = "https://youtu.be/-FiXO2K4xeA"
DURACAO = "4min46s, 1080p"

# números da execução final no Colab (preenchidos a partir do notebook executado)
R = {
    "v30": "0,920", "t30": "0,739", "tt30": "0,439", "tr30": "2,6 min",
    "v60": "0,995", "t60": "<strong>0,811</strong>", "tt60": "<strong>0,569</strong>", "tr60": "2,2 min",
    "a_map": "<strong>0,798</strong>", "a_acc": "87,5%", "a_tr": "~2 min", "a_inf": "~21 ms",
    "b_map": "0,665", "b_acc": "87,5%", "b_inf": "~21 ms",
    "c_acc": "<strong>100%</strong>", "c_tr": "~30 s", "c_inf": "~13 ms",
    "conclusao": ("<strong>Conclusão.</strong> Com 60 épocas o detector melhorou em todas as medidas de teste e "
                  "não houve overfitting. O ajuste fino com 64 fotos rendeu 13 pontos de mAP@0.5 sobre a YOLO "
                  "pronta do COCO, que ainda é melhor em cenas de rebanho; a CNN do zero acerta todas as "
                  "classificações, mas não localiza o objeto. Recomendação: YOLOv5 customizado de 60 épocas, "
                  "recebendo imagens de um ESP32-CAM por Wi-Fi."),
}

HTML = """
<!doctype html><meta charset="utf-8">
<style>
  @page {{ size: A4; margin: 18mm 17mm; }}
  * {{ box-sizing: border-box; }}
  body {{ font: 10.5pt/1.5 -apple-system, "Helvetica Neue", Arial, sans-serif; color: #1a1d21; margin: 0; }}
  /* sem ligaduras: "fi" viraria U+FB01 e o link copiado do PDF deixaria de funcionar */
  a, code, .link {{ font-variant-ligatures: none; -webkit-font-feature-settings: "liga" 0; }}
  .capa {{ border-bottom: 3px solid #ed145b; padding-bottom: 12px; margin-bottom: 18px; }}
  .fase {{ color: #ed145b; font-size: 9pt; letter-spacing: .13em; text-transform: uppercase; font-weight: 700; }}
  h1 {{ font-size: 18pt; margin: 6px 0 3px; letter-spacing: -.3px; }}
  .sub {{ color: #5c6570; font-size: 10.5pt; }}
  h2 {{ font-size: 12pt; margin: 20px 0 8px; padding-bottom: 4px; border-bottom: 1px solid #dde1e6; }}
  table {{ width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 9.5pt; }}
  th, td {{ border: 1px solid #dde1e6; padding: 5px 9px; text-align: left; }}
  th {{ background: #f5f6f8; font-weight: 600; }}
  .link {{ background: #f5f6f8; border: 1px solid #dde1e6; border-radius: 6px; padding: 10px 14px; margin: 8px 0; }}
  .link .rot {{ font-size: 8pt; text-transform: uppercase; letter-spacing: .09em; color: #5c6570; font-weight: 700; margin-bottom: 2px; }}
  .link a {{ color: #0b5cd5; font-size: 10.5pt; word-break: break-all; text-decoration: none; font-weight: 600; }}
  .link .obs {{ font-size: 8.5pt; color: #5c6570; margin-top: 2px; }}
  ul {{ margin: 6px 0 6px 17px; padding: 0; }} li {{ margin-bottom: 3px; }}
  .destaque {{ background: #fff8e6; border-left: 3px solid #e8a317; padding: 9px 13px; margin: 10px 0; font-size: 9.8pt; }}
  .rodape {{ margin-top: 20px; padding-top: 9px; border-top: 1px solid #dde1e6; font-size: 8pt; color: #8b939c; }}
</style>

<div class="capa">
  <div class="fase">FIAP · Inteligência Artificial · 1TIAOA · Fase 6</div>
  <h1>Cap 1 — O despertar da Rede Neural</h1>
  <div class="sub">FarmTech Solutions · visão computacional com YOLO · Entregas 1 e 2</div>
</div>

<h2>Grupo 51</h2>
<table>
  <tr><th>Integrante</th><th>RM</th></tr>
  <tr><td>Douglas Felicio da Silva</td><td>RM572312</td></tr>
  <tr><td>Gabriel Kaique Portel Silva</td><td>RM572500</td></tr>
  <tr><td>Richard Wrobel dos Santos</td><td>RM573998</td></tr>
</table>

<h2>Links da entrega</h2>
<div class="link"><div class="rot">Repositório GitHub — público</div>
  <a href="{REPO}">{REPO}</a>
  <div class="obs">README no template FIAP, base de imagens, rótulos do Make Sense e scripts.</div></div>
<div class="link"><div class="rot">Notebook (executado no Google Colab)</div>
  <a href="{NOTEBOOK}">{NOTEBOOK}</a>
  <div class="obs">Abrir direto no Colab: <a href="{COLAB}" style="font-size:9pt">{COLAB}</a></div></div>
<div class="link"><div class="rot">Vídeo de demonstração — YouTube, não listado</div>
  <a href="{VIDEO}">{VIDEO}</a>
  <div class="obs">{DURACAO} · o mesmo link está no README</div></div>

<h2>Entrega 1 — YOLOv5 customizado</h2>
<ul>
  <li><strong>Base:</strong> 80 imagens do Open Images V7 (CC BY 2.0) — 40 de <strong>gado bovino</strong> e 40 de <strong>maçã</strong>,
      32/4/4 por classe em treino/validação/teste, organizadas no Google Drive</li>
  <li><strong>Rotulagem:</strong> 101 objetos marcados no <strong>Make Sense</strong> e exportados em formato YOLO; rótulos salvos no Drive</li>
  <li><strong>Treino no Colab (GPU T4)</strong> a partir do YOLOv5s, em duas simulações: <strong>30</strong> e <strong>60 épocas</strong></li>
  <li><strong>Prints do teste</strong> em <code>yolov5/runs/detect/exp</code> (30 ép.) e <code>exp2</code> (60 ép.)</li>
</ul>
<table>
  <tr><th>Simulação</th><th>mAP@0.5 validação</th><th>mAP@0.5 teste</th><th>mAP@0.5:0.95 teste</th><th>Tempo de treino</th></tr>
  <tr><td>30 épocas</td><td>{v30}</td><td>{t30}</td><td>{tt30}</td><td>{tr30}</td></tr>
  <tr><td>60 épocas</td><td>{v60}</td><td>{t60}</td><td>{tt60}</td><td>{tr60}</td></tr>
</table>

<h2>Entrega 2 — Comparação das abordagens (mesmas 8 imagens de teste)</h2>
<table>
  <tr><th>Abordagem</th><th>mAP@0.5</th><th>Acurácia (imagem)</th><th>Treino</th><th>Inferência</th></tr>
  <tr><td>A · YOLOv5 customizado (60 épocas)</td><td>{a_map}</td><td>{a_acc}</td><td>{a_tr}</td><td>{a_inf}</td></tr>
  <tr><td>B · YOLO tradicional (COCO, sem treino)</td><td>{b_map}</td><td>{b_acc}</td><td>—</td><td>{b_inf}</td></tr>
  <tr><td>C · CNN treinada do zero</td><td>não detecta</td><td>{c_acc}</td><td>{c_tr}</td><td>{c_inf}</td></tr>
</table>
<div class="destaque">{conclusao}</div>

<div class="rodape">Atividade em grupo · prazo 13/10/2026 · nenhum commit após a data de entrega.</div>
"""


def gerar() -> Path:
    assert R and "__" not in VIDEO, "preencha R e VIDEO antes de gerar"
    destino = BASE / "FarmTech_Fase6_Cap1_Entrega_Grupo51.pdf"
    origem = BASE / "video" / ".entrega.html"
    origem.write_text(HTML.format(REPO=REPO, NOTEBOOK=NOTEBOOK, COLAB=COLAB, VIDEO=VIDEO,
                                  DURACAO=DURACAO, **R), encoding="utf-8")
    with sync_playwright() as p:
        navegador = p.chromium.launch()
        pagina = navegador.new_page()
        pagina.goto(origem.as_uri())
        pagina.wait_for_timeout(800)
        pagina.pdf(path=str(destino), format="A4", print_background=True)
        navegador.close()
    origem.unlink(missing_ok=True)
    return destino


if __name__ == "__main__":
    d = gerar()
    print(f"{d.name}: {d.stat().st_size / 1024:.0f} KB")
