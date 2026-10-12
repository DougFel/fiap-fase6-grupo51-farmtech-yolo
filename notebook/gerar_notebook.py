"""
Monta o notebook da entrega (Colab) a partir de blocos de markdown e código.

O notebook é escrito aqui, e não editado à mão no Colab, para que o texto e o
código fiquem versionados e revisáveis. Uso:

    python gerar_notebook.py            # notebook da entrega (30 e 60 épocas)
    python gerar_notebook.py --rapido   # versão de fumaça para testar localmente

A versão --rapido troca apenas os hiperparâmetros de duração (épocas e tamanho de
imagem) para que o pipeline inteiro rode em poucos minutos sem GPU.
"""
import argparse
from pathlib import Path

import nbformat as nbf

from textos_analise import AVALIACAO_CRITICA, CONCLUSOES_ENTREGA1, FECHAMENTO

AQUI = Path(__file__).resolve().parent
NOME = "DouglasFelicioDaSilva_rm572312_pbl_fase6.ipynb"
REPO = "https://github.com/DougFel/fiap-fase6-grupo51-farmtech-yolo"
COMMIT_YOLO = "f9578796b8a64889aa9e66dc06914d19f40d338e"


def md(texto: str):
    return nbf.v4.new_markdown_cell(texto.strip("\n"))


def py(texto: str):
    return nbf.v4.new_code_cell(texto.strip("\n"))


def celulas(rapido: bool) -> list:
    ep_a, ep_b = (1, 2) if rapido else (30, 60)
    img = 320 if rapido else 640
    ep_cnn = 3 if rapido else 150
    c = []

    # ------------------------------------------------------------------ capa
    c.append(md(f"""
# FarmTech Solutions — Visão Computacional com YOLO
### Fase 6 · Cap 1 — O despertar da Rede Neural

**FIAP · Inteligência Artificial · 1TIAOA · 2026/1 — Grupo 51**

| Integrante | RM |
|---|---|
| Douglas Felicio da Silva | RM572312 |
| Gabriel Kaique Portel Silva | RM572500 |
| Richard Wrobel dos Santos | RM573998 |

Repositório: <{REPO}>

---

**O pedido do cliente.** A FarmTech ampliou sua carteira para saúde animal e
segurança de propriedades rurais. Um cliente quer ver, na prática, como um
sistema de visão computacional reconhece objetos no campo — e com que confiança.

**O que este notebook entrega**

| Parte | Conteúdo | Seções |
|---|---|---|
| **Entrega 1** | YOLOv5 customizado para dois objetos (gado e maçã), treinado com **30** e **60** épocas, validado e testado | 1 a 6 |
| **Entrega 2** | Comparação com a **YOLO tradicional** (pré-treinada no COCO) e com uma **CNN treinada do zero** | 7 a 10 |
| Fechamento | Pontos fortes, limitações e próximos passos | 11 |

> **Como executar:** `Ambiente de execução → Alterar o tipo de ambiente → GPU (T4)`
> e depois `Executar tudo`. O notebook monta o **seu** Google Drive e, se a base
> ainda não estiver lá, copia as imagens e os rótulos do repositório para
> `MyDrive/FarmTech_Fase6/`. Tempo total com GPU T4: cerca de 15 minutos.
"""))

    # ------------------------------------------------------------- 1. cenário
    c.append(md("""
## 1. Cenário e escolha dos objetos

Escolhemos dois objetos **bem diferentes entre si**, como pede o enunciado, e
ambos ligados ao negócio da FarmTech:

| Classe | Objeto | Por que interessa ao cliente |
|---|---|---|
| `0 · gado` | gado bovino (vacas, bois, bezerros) | contagem de rebanho, saúde animal, cerca virtual |
| `1 · maca` | maçã | inspeção de pomar e de produção na esteira |

Os dois contrastam em quase tudo que importa para uma rede de detecção:
**escala** (um animal de 600 kg contra um fruto de 200 g), **forma** (silhueta
articulada contra contorno arredondado), **textura** (pelagem contra casca lisa) e
**cenário** (pasto e curral contra mesa, galho e estúdio).

**Origem das imagens.** As 80 fotos vêm do **Open Images V7** (Google), todas com
licença **Creative Commons BY 2.0**; autoria e link de cada uma estão em
`dataset/fontes.csv`. Partimos de um filtro automático (foto real, até 3 objetos,
objeto ocupando ao menos 4% da imagem) e depois fizemos **curadoria visual**: a
classe "Cattle" do Open Images mistura iaques, bisões e búfalos, e "Apple" inclui
tortas e fatias — ficamos só com gado bovino doméstico e maçãs inteiras.

**Rotulagem.** As caixas foram desenhadas no **Make Sense** (makesense.ai), uma
caixa por objeto, e exportadas em formato YOLO (um `.txt` por imagem). O pacote
original exportado está em `dataset/makesense/` e as caixas, em `dataset/labels/`.

![Rotulagem no Make Sense](https://raw.githubusercontent.com/DougFel/fiap-fase6-grupo51-farmtech-yolo/main/assets/makesense_rotulagem_gado.png)
*Projeto `farmtech-gado-maca` no Make Sense: as 80 imagens carregadas (✓ = rotulada) e duas
vacas marcadas com a classe `gado`.*
"""))

    # ------------------------------------------------------------ 2. ambiente
    c.append(md("""
## 2. Ambiente: GPU, Google Drive e YOLOv5

A célula abaixo detecta se estamos no Colab, monta o Google Drive e define as
pastas do projeto. Fora do Colab (execução local), uma pasta `drive_local/` ao lado
do repositório faz o papel do Drive, para que o mesmo código rode nos dois lugares.
"""))
    c.append(py("""
import os, sys, time, json, shutil, subprocess, random      # utilitários da biblioteca padrão
from pathlib import Path                                   # caminhos independentes de sistema operacional

EM_COLAB = 'google.colab' in sys.modules                   # True quando o notebook roda no Colab
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'                   # silencia avisos do TensorFlow (também nos subprocessos)

if EM_COLAB:
    from google.colab import drive                         # integração oficial do Colab com o Drive
    drive.mount('/content/drive')                          # pede autorização e monta o Drive do usuário
    RAIZ_DRIVE = Path('/content/drive/MyDrive')            # "Meu Drive"
    TRABALHO = Path('/content')                            # disco local da máquina do Colab (rápido)
else:
    RAIZ_DRIVE = Path.cwd().parent / 'drive_local'         # simula o Drive na execução local
    TRABALHO = Path.cwd().parent / '_trabalho'

PROJETO = RAIZ_DRIVE / 'FarmTech_Fase6'                    # pasta do projeto dentro do Drive
DATASET = PROJETO / 'dataset'                              # imagens + rótulos (treino, validação, teste)
RESULTADOS = PROJETO / 'resultados'                        # tudo que o notebook produz é salvo aqui
for pasta in (PROJETO, RESULTADOS, TRABALHO):
    pasta.mkdir(parents=True, exist_ok=True)

print('Executando no Colab' if EM_COLAB else 'Executando localmente')
print('Pasta do projeto:', PROJETO)
"""))
    c.append(py("""
import torch                                               # PyTorch: base do YOLOv5

GPU = torch.cuda.is_available()                            # há GPU CUDA disponível?
DISPOSITIVO = '0' if GPU else 'cpu'                        # formato esperado pelos scripts do YOLOv5
print('PyTorch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if GPU else 'não (CPU)')
if not GPU:
    print('Atenção: sem GPU o treino é ~100x mais lento. No Colab, ative a GPU T4.')
"""))

    c.append(md("""
### 2.1 Base de imagens no Google Drive

Se a base ainda não estiver no Drive, ela é copiada do repositório do grupo. A
estrutura segue o padrão YOLO, com uma pasta por etapa:

```
MyDrive/FarmTech_Fase6/dataset/
├── images/  train/ (64)   val/ (8)   test/ (8)
├── labels/  train/ (64)   val/ (8)   test/ (8)    ← caixas do Make Sense (.txt)
├── makesense/  pacote exportado do Make Sense + lista de rótulos
└── fontes.csv  autoria e licença de cada imagem
```
"""))
    c.append(py(f"""
REPO_URL = '{REPO}'                     # repositório público do grupo

if not (DATASET / 'images').exists():                      # base ausente no Drive → copiar do repositório
    if EM_COLAB:
        destino = TRABALHO / 'repo'
        if not destino.exists():
            subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, str(destino)], check=True)
        origem = destino / 'dataset'
    else:
        origem = Path.cwd().parent / 'dataset'             # local: a base já está no repositório
    shutil.copytree(origem, DATASET)                       # grava imagens e rótulos no Drive
    print('Base copiada para o Drive.')
else:
    print('Base já estava no Drive.')

# Conferência: quantas imagens e rótulos existem em cada etapa e de cada classe
for etapa in ('train', 'val', 'test'):
    imgs = sorted((DATASET / 'images' / etapa).glob('*.jpg'))
    rots = sorted((DATASET / 'labels' / etapa).glob('*.txt'))
    gado = sum(p.name.startswith('gado') for p in imgs)
    print(f'{{etapa:>5}}: {{len(imgs):2d}} imagens ({{gado}} gado, {{len(imgs) - gado}} maçã) | {{len(rots):2d}} arquivos de rótulo')
"""))

    c.append(md("""
### 2.2 Conferência visual da rotulagem

Antes de treinar, desenhamos as caixas do Make Sense sobre algumas imagens de
treino. É a forma mais barata de pegar erro de rotulagem — o tipo de erro que um
modelo aprende e repete.
"""))
    c.append(py("""
import numpy as np                                          # vetores e estatísticas
import pandas as pd                                         # tabelas
import io                                                   # buffer em memória para as figuras
import matplotlib.pyplot as plt                             # gráficos
import matplotlib.patches as patches                        # retângulos sobre as imagens
from PIL import Image                                       # leitura de imagens

from IPython.display import Image as ImagemPNG

def mostrar(fig, dpi=100):
    \"\"\"Exibe a figura como PNG. Independe do backend: o YOLOv5 troca o do matplotlib para 'Agg' ao ser importado.\"\"\"
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    display(ImagemPNG(buf.getvalue()))

NOMES = {0: 'gado', 1: 'maca'}                              # mesma ordem dos rótulos no Make Sense
CORES = {0: '#e4572e', 1: '#2e86ab'}                        # uma cor por classe

def ler_rotulos(txt: Path):
    \"\"\"Lê um arquivo YOLO: classe, centro x, centro y, largura, altura (normalizados 0-1).\"\"\"
    if not txt.exists():
        return []
    return [list(map(float, linha.split())) for linha in txt.read_text().splitlines() if linha.strip()]

def desenhar(ax, img_path: Path, rotulos, titulo=''):
    \"\"\"Mostra a imagem com as caixas convertidas de YOLO (normalizado) para pixels.\"\"\"
    img = Image.open(img_path); w, h = img.size
    ax.imshow(img); ax.set_axis_off(); ax.set_title(titulo, fontsize=9)
    for cls, xc, yc, bw, bh in rotulos:
        x0, y0 = (xc - bw / 2) * w, (yc - bh / 2) * h       # canto superior esquerdo
        ax.add_patch(patches.Rectangle((x0, y0), bw * w, bh * h, fill=False, lw=2, color=CORES[int(cls)]))
        ax.text(x0, y0, NOMES[int(cls)], color='white', fontsize=8, backgroundcolor=CORES[int(cls)])

amostra = [f'gado_train_{i:02d}.jpg' for i in (1, 5, 7, 22)] + [f'maca_train_{i:02d}.jpg' for i in (7, 12, 17, 32)]
fig, eixos = plt.subplots(2, 4, figsize=(16, 7))
for ax, nome in zip(eixos.ravel(), amostra):
    p = DATASET / 'images' / 'train' / nome
    desenhar(ax, p, ler_rotulos(DATASET / 'labels' / 'train' / p.with_suffix('.txt').name), nome)
plt.suptitle('Rótulos feitos no Make Sense (amostra do treino)', fontsize=13)
plt.tight_layout(); mostrar(fig)
"""))
    c.append(py("""
# Estatísticas da base: objetos por imagem e tamanho das caixas (fração da área da imagem)
linhas = []
for etapa in ('train', 'val', 'test'):
    for txt in (DATASET / 'labels' / etapa).glob('*.txt'):
        for cls, xc, yc, bw, bh in ler_rotulos(txt):
            linhas.append({'etapa': etapa, 'classe': NOMES[int(cls)], 'imagem': txt.stem, 'area_relativa': bw * bh})
caixas = pd.DataFrame(linhas)

resumo = caixas.groupby(['etapa', 'classe']).agg(imagens=('imagem', 'nunique'), objetos=('imagem', 'size'),
                                                  area_media=('area_relativa', 'mean')).round(3)
display(resumo)
print(f"Total: {caixas.imagem.nunique()} imagens, {len(caixas)} objetos rotulados")
"""))
    c.append(md("""
**Leitura da base.** Cada imagem tem de 1 a 3 objetos, e os objetos são grandes
(em média ocupam mais da metade da foto). É um cenário favorável para o YOLO: com
só 32 imagens por classe, objetos pequenos ou muito sobrepostos seriam o primeiro
ponto de falha. Ao mesmo tempo, isso é uma **limitação** a declarar ao cliente —
voltamos a ela na seção 11.
"""))

    c.append(md("""
### 2.3 YOLOv5

Usamos o repositório oficial da Ultralytics, **fixado num commit** para que o
resultado seja reprodutível mesmo que o repositório mude depois da entrega.
"""))
    c.append(py(f"""
YOLO_DIR = TRABALHO / 'yolov5'
COMMIT = '{COMMIT_YOLO}'                         # versão do YOLOv5 usada no projeto

if not YOLO_DIR.exists():
    subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5', str(YOLO_DIR)], check=True)
    subprocess.run(['git', '-C', str(YOLO_DIR), 'checkout', '-q', COMMIT], check=True)
if EM_COLAB:                                                # no Colab, instala as dependências do YOLOv5
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(YOLO_DIR / 'requirements.txt')], check=True)
sys.path.insert(0, str(YOLO_DIR))                           # permite importar val.py e os utilitários
from utils.downloads import attempt_download                # utilitário do próprio YOLOv5
attempt_download(YOLO_DIR / 'yolov5s.pt')                   # baixa os pesos COCO agora: o download não entra no tempo de treino
print('YOLOv5 em', YOLO_DIR, '| commit', subprocess.run(['git', '-C', str(YOLO_DIR), 'rev-parse', '--short', 'HEAD'],
      capture_output=True, text=True).stdout.strip())
"""))
    c.append(py(f"""
# Arquivo de configuração da base: onde estão as imagens e quais são as classes
DATA_YAML = PROJETO / 'farmtech.yaml'
DATA_YAML.write_text(f\"\"\"path: {{DATASET}}
train: images/train
val: images/val
test: images/test
names:
  0: gado
  1: maca
\"\"\")
print(DATA_YAML.read_text())

IMG = {img}                                                  # tamanho de entrada da rede (pixels)
"""))

    # ---------------------------------------------------------- 3. treino
    c.append(md(f"""
## 3. Treinamento — duas simulações: {ep_a} e {ep_b} épocas

Partimos dos pesos **`yolov5s.pt`** (pré-treinados no COCO) e ajustamos a rede às
nossas duas classes — o caminho recomendado no capítulo 3 para bases pequenas.
As duas simulações são **idênticas em tudo, menos no número de épocas**, para que
a diferença de resultado possa ser atribuída só a esse parâmetro:

| Parâmetro | Valor | Motivo |
|---|---|---|
| Modelo base | YOLOv5s (7 M parâmetros) | leve, treina rápido na T4 e roda em tempo real |
| Imagem | {img} px | tamanho nativo do YOLOv5 |
| Lote (batch) | 16 | cabe na memória da T4 com folga |
| Semente | 0 (padrão do YOLOv5) | mesma inicialização nas duas simulações |
| Épocas | **{ep_a}** e **{ep_b}** | bem diferentes, como pede o enunciado |

O treino usa a pasta `train` e, ao fim de cada época, avalia na pasta `val`
(validação). O melhor peso (`best.pt`) é escolhido pela métrica de validação.
"""))
    c.append(py(f"""
def rodar(cmd, cwd=YOLO_DIR, linhas_finais=12):
    \"\"\"Executa um script do YOLOv5, mostra o fim da saída e devolve o tempo gasto (s).\"\"\"
    inicio = time.time()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    ruido = ('absl', 'cudart_stub', 'cuda drivers', 'B/s]', 'it/s]', 'Downloading')  # avisos e barras de download
    saida = [l for l in (proc.stdout + proc.stderr).splitlines() if l.strip() and not any(r in l for r in ruido)]
    print('\\n'.join(saida[-linhas_finais:]))
    if proc.returncode != 0:
        raise RuntimeError('falhou: ' + ' '.join(map(str, cmd)))
    return time.time() - inicio

EXPERIMENTOS = {{'30 épocas': {ep_a}, '60 épocas': {ep_b}}}  # nome do experimento → épocas
tempos_treino = {{}}

for nome, epocas in EXPERIMENTOS.items():
    print(f'===== Treinando: {{nome}} =====')
    tempos_treino[nome] = rodar([
        sys.executable, 'train.py',
        '--img', str(IMG), '--batch', '16', '--epochs', str(epocas),
        '--data', str(DATA_YAML), '--weights', 'yolov5s.pt',
        '--project', str(YOLO_DIR / 'runs' / 'train'), '--name', f'farmtech_{{epocas}}ep', '--exist-ok',
        '--cache', 'ram', '--device', DISPOSITIVO,
    ])
    print(f'Tempo de treino ({{nome}}): {{tempos_treino[nome] / 60:.1f}} min\\n')

PESOS = {{nome: YOLO_DIR / 'runs' / 'train' / f'farmtech_{{ep}}ep' / 'weights' / 'best.pt' for nome, ep in EXPERIMENTOS.items()}}
"""))

    c.append(md("""
### 3.1 Curvas de treino: erro e acerto por época

O `results.csv` de cada treino registra, época a época, as **perdas** (o "erro"
que a rede minimiza) e as métricas na validação. As três perdas do YOLO medem
coisas diferentes:

- **box** — quão longe a caixa prevista está da caixa verdadeira;
- **obj** — se a rede acerta *onde há objeto* (objectness);
- **cls** — se acerta *qual* objeto é (gado ou maçã).
"""))
    c.append(py("""
def historico(nome):
    \"\"\"Lê o results.csv de um treino (o YOLOv5 grava os nomes das colunas com espaços).\"\"\"
    df = pd.read_csv(PESOS[nome].parents[1] / 'results.csv')
    df.columns = [col.strip() for col in df.columns]
    return df

hist = {nome: historico(nome) for nome in EXPERIMENTOS}

graficos = [('train/box_loss', 'val/box_loss', 'Perda de caixa (box)'),
            ('train/obj_loss', 'val/obj_loss', 'Perda de objeto (obj)'),
            ('train/cls_loss', 'val/cls_loss', 'Perda de classe (cls)'),
            ('metrics/precision', None, 'Precisão (validação)'),
            ('metrics/recall', None, 'Recall (validação)'),
            ('metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'mAP@0.5 (cheio) e mAP@0.5:0.95 (tracejado)')]
fig, eixos = plt.subplots(2, 3, figsize=(17, 8.5))
for ax, (col1, col2, titulo) in zip(eixos.ravel(), graficos):
    for nome, cor in zip(EXPERIMENTOS, ('#e4572e', '#2e86ab')):
        df = hist[nome]
        ax.plot(df.epoch + 1, df[col1], color=cor, label=f'{nome}' + (' — treino' if 'loss' in col1 else ''))
        if col2:
            ax.plot(df.epoch + 1, df[col2], color=cor, ls='--', label=f'{nome} — validação' if 'loss' in col2 else None)
    ax.set_title(titulo); ax.set_xlabel('época'); ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(RESULTADOS / 'curvas_treino.png', dpi=120); mostrar(fig)
"""))
    c.append(md("""
**Como ler as curvas.** As linhas cheias de perda caindo mostram que a rede está
aprendendo; as tracejadas (validação) mostram se esse aprendizado vale para
imagens que ela não viu. Se a perda de treino continuasse caindo enquanto a de
validação subisse, seria o sinal clássico de *overfitting*. A comparação das
duas simulações está na seção 5, depois da avaliação no teste.
"""))

    # --------------------------------------------------- 4. validação e teste
    c.append(md("""
## 4. Validação e teste

Avaliamos o `best.pt` de cada simulação em duas bases que **não participaram do
ajuste dos pesos**:

- **validação** (8 imagens) — usada pelo YOLO para escolher o melhor peso;
- **teste** (8 imagens) — nunca vista em nenhuma decisão de treino; é a medida
  honesta de como o modelo se comportaria no cliente.

Métricas: **precisão** (das detecções feitas, quantas estão certas), **recall**
(dos objetos reais, quantos foram achados), **mAP@0.5** (precisão média com caixa
aceita se IoU ≥ 0,5) e **mAP@0.5:0.95** (média em IoU de 0,5 a 0,95, mais exigente).
"""))
    c.append(py("""
import val as yolo_val                                      # módulo de avaliação do próprio YOLOv5

def avaliar(nome, etapa):
    \"\"\"Roda o val.py do YOLOv5 e devolve as métricas gerais e o tempo de inferência.\"\"\"
    (p, r, map50, map5095, *_), _, tempos = yolo_val.run(
        data=str(DATA_YAML), weights=str(PESOS[nome]), task=etapa, imgsz=IMG, batch_size=8,
        device=DISPOSITIVO, project=str(YOLO_DIR / 'runs' / 'val'), name=f"{nome.split()[0]}ep_{etapa}",
        exist_ok=True, verbose=True, plots=True)
    return {'simulação': nome, 'base': etapa, 'precisão': p, 'recall': r, 'mAP@0.5': map50,
            'mAP@0.5:0.95': map5095, 'inferência (ms/img)': tempos[1]}

avaliar(next(iter(EXPERIMENTOS)), 'val')                  # aquecimento da GPU: a 1ª chamada carrega kernels CUDA e distorce o tempo
metricas = pd.DataFrame([avaliar(nome, etapa) for nome in EXPERIMENTOS for etapa in ('val', 'test')])
display(metricas.round(3))
"""))

    c.append(md("""
### 4.1 Prints do teste — o que o cliente vai ver

O `detect.py` processa as 8 imagens de teste e grava as imagens com as caixas e a
confiança de cada detecção. A saída fica em `yolov5/runs/detect/expX`, onde X
aumenta a cada execução — aqui, `exp` para 30 épocas e `exp2` para 60.
"""))
    c.append(py("""
pastas_detect = {}
for nome in EXPERIMENTOS:
    antes = set((YOLO_DIR / 'runs' / 'detect').glob('exp*'))
    rodar([sys.executable, 'detect.py', '--weights', str(PESOS[nome]), '--source', str(DATASET / 'images' / 'test'),
           '--img', str(IMG), '--conf-thres', '0.25', '--device', DISPOSITIVO], linhas_finais=3)
    nova = sorted(set((YOLO_DIR / 'runs' / 'detect').glob('exp*')) - antes, key=lambda p: p.stat().st_mtime)[-1]
    pastas_detect[nome] = nova                              # guarda qual expX pertence a qual simulação
    print(f'{nome}: resultados em {nova.relative_to(TRABALHO)}\\n')
"""))
    c.append(py("""
def galeria(pasta, titulo):
    \"\"\"Mostra as 8 imagens de teste processadas pelo modelo.\"\"\"
    imagens = sorted(pasta.glob('*.jpg'))
    fig, eixos = plt.subplots(2, 4, figsize=(18, 8.5))
    for ax, p in zip(eixos.ravel(), imagens):
        ax.imshow(Image.open(p)); ax.set_axis_off(); ax.set_title(p.name, fontsize=9)
    plt.suptitle(titulo, fontsize=14); plt.tight_layout()
    plt.savefig(RESULTADOS / f'teste_{pasta.name}.jpg', dpi=110); mostrar(fig)

for nome, pasta in pastas_detect.items():
    galeria(pasta, f'Imagens de teste processadas — YOLOv5 customizado, {nome} ({pasta.name})')
"""))

    # --------------------------------------------------------- 5. comparação
    c.append(md("""
## 5. 30 x 60 épocas: acurácia, erro e desempenho

Juntamos numa tabela o que o enunciado pede para comparar: **acurácia** (precisão,
recall, mAP), **erro** (perdas finais de treino e validação) e **desempenho**
(tempo de treino e de inferência).
"""))
    c.append(py("""
linhas = []
for nome in EXPERIMENTOS:
    fim = hist[nome].iloc[-1]                                # última época de cada treino
    teste = metricas[(metricas['simulação'] == nome) & (metricas.base == 'test')].iloc[0]
    validacao = metricas[(metricas['simulação'] == nome) & (metricas.base == 'val')].iloc[0]
    linhas.append({
        'simulação': nome,
        'mAP@0.5 val': validacao['mAP@0.5'], 'mAP@0.5 teste': teste['mAP@0.5'],
        'mAP@0.5:0.95 teste': teste['mAP@0.5:0.95'],
        'precisão teste': teste['precisão'], 'recall teste': teste['recall'],
        'perda total treino': fim['train/box_loss'] + fim['train/obj_loss'] + fim['train/cls_loss'],
        'perda total val': fim['val/box_loss'] + fim['val/obj_loss'] + fim['val/cls_loss'],
        'tempo de treino (min)': tempos_treino[nome] / 60,
        'inferência (ms/img)': teste['inferência (ms/img)'],
    })
comparacao_epocas = pd.DataFrame(linhas).set_index('simulação')
display(comparacao_epocas.round(3).T)
comparacao_epocas.to_csv(RESULTADOS / 'comparacao_30x60.csv')
"""))
    c.append(md("### Conclusões da Entrega 1\n\n" + CONCLUSOES_ENTREGA1))

    # ---------------------------------------------------------- Entrega 2
    c.append(md("""
---
# Entrega 2 — A YOLO customizada contra as alternativas

Avaliamos três abordagens sobre **a mesma base de teste** (as 8 imagens que
nenhum modelo usou para aprender):

| # | Abordagem | O que faz | Treino nosso |
|---|---|---|---|
| A | **YOLOv5 customizado** (Entrega 1) | detecta e classifica gado e maçã | sim (ajuste fino) |
| B | **YOLO tradicional** | YOLOv5s com os pesos originais do COCO, sem nenhum ajuste | não |
| C | **CNN treinada do zero** | classifica a imagem inteira em gado ou maçã | sim (do zero) |

Para comparar detectores e classificador de forma justa, usamos duas medidas:

1. **mAP@0.5 de detecção** — só para A e B, que localizam objetos;
2. **acurácia de classificação da imagem** — para as três: nos detectores, a
   classe da imagem é a da detecção mais confiante.
"""))

    c.append(md("""
## 7. YOLO tradicional (COCO, sem customização)

O COCO tem 80 classes, entre elas **`cow`** (19) e **`apple`** (47). Rodamos o
YOLOv5s original só com essas duas classes e mapeamos `cow → gado` e
`apple → maca`. Assim medimos o que o cliente teria **sem treinar nada**.
"""))
    c.append(py("""
COCO_PARA_NOSSO = {19: 0, 47: 1}                            # cow → gado, apple → maca

antes = set((YOLO_DIR / 'runs' / 'detect').glob('exp*'))
rodar([sys.executable, 'detect.py', '--weights', 'yolov5s.pt', '--source', str(DATASET / 'images' / 'test'),
       '--img', str(IMG), '--conf-thres', '0.25', '--classes', '19', '47', '--device', DISPOSITIVO], linhas_finais=3)
pasta_coco = sorted(set((YOLO_DIR / 'runs' / 'detect').glob('exp*')) - antes, key=lambda p: p.stat().st_mtime)[-1]
galeria(pasta_coco, f'Imagens de teste — YOLO tradicional, pesos COCO ({pasta_coco.name})')
"""))

    c.append(md("""
### 7.1 Avaliação padronizada dos detectores

Para comparar A e B com a mesma régua, carregamos cada modelo pelo
`torch.hub`, rodamos as 8 imagens de teste e calculamos as métricas com uma
função própria (IoU ≥ 0,5, AP com interpolação em todos os pontos, como no VOC).
Medimos também o **tempo de inferência por imagem**, depois de um aquecimento.
"""))
    c.append(py("""
IMAGENS_TESTE = sorted((DATASET / 'images' / 'test').glob('*.jpg'))
CLASSE_REAL = {p.name: 0 if p.name.startswith('gado') else 1 for p in IMAGENS_TESTE}

def caixas_reais(img_path):
    \"\"\"Caixas do Make Sense em pixels: [classe, x1, y1, x2, y2].\"\"\"
    w, h = Image.open(img_path).size
    out = []
    for cls, xc, yc, bw, bh in ler_rotulos(DATASET / 'labels' / 'test' / img_path.with_suffix('.txt').name):
        out.append([int(cls), (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
    return out

def iou(a, b):
    \"\"\"Interseção sobre união entre duas caixas [x1, y1, x2, y2].\"\"\"
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

def ap_voc(recall, precisao):
    \"\"\"Área sob a curva precisão-recall com interpolação em todos os pontos.\"\"\"
    r = np.concatenate(([0.], recall, [1.])); p = np.concatenate(([1.], precisao, [0.]))
    p = np.flip(np.maximum.accumulate(np.flip(p)))          # envelope monotônico da precisão
    i = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[i + 1] - r[i]) * p[i + 1]))

def avaliar_detector(nome, pesos, mapa_classes=None, conf=0.25):
    \"\"\"Roda um detector no teste e devolve métricas de detecção e de classificação da imagem.\"\"\"
    modelo = torch.hub.load(str(YOLO_DIR), 'custom', path=str(pesos), source='local', verbose=False)
    modelo.conf = 0.001                                      # guarda tudo para calcular o AP; o limiar entra depois
    if mapa_classes:
        modelo.classes = list(mapa_classes)                 # só as classes de interesse (COCO)
    modelo(str(IMAGENS_TESTE[0]), size=IMG)                  # aquecimento (não entra na medição de tempo)

    deteccoes, tempos = [], []
    for p in IMAGENS_TESTE:
        if GPU: torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = modelo(str(p), size=IMG).xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, classe]
        if GPU: torch.cuda.synchronize()
        tempos.append((time.perf_counter() - t0) * 1000)
        for x1, y1, x2, y2, cf, cl in pred:
            cl = mapa_classes[int(cl)] if mapa_classes else int(cl)
            deteccoes.append((p.name, cl, float(cf), [x1, y1, x2, y2]))

    # --- detecção: AP por classe com IoU >= 0,5
    aps, tp_total, fp_total, n_reais = [], 0, 0, 0
    for cls in (0, 1):
        reais = {p.name: [c[1:] for c in caixas_reais(p) if c[0] == cls] for p in IMAGENS_TESTE}
        usados = {k: [False] * len(v) for k, v in reais.items()}
        n_cls = sum(len(v) for v in reais.values()); n_reais += n_cls
        dets = sorted([d for d in deteccoes if d[1] == cls], key=lambda d: -d[2])
        tp = []
        for img, _, cf, caixa in dets:
            ious = [iou(caixa, g) for g in reais[img]]
            j = int(np.argmax(ious)) if ious else -1
            acerto = j >= 0 and ious[j] >= 0.5 and not usados[img][j]
            if acerto: usados[img][j] = True
            tp.append(acerto)
            if cf >= conf: tp_total += acerto; fp_total += (not acerto)
        tp = np.array(tp, dtype=float)
        acum_tp, acum_fp = np.cumsum(tp), np.cumsum(1 - tp)
        aps.append(ap_voc(acum_tp / max(n_cls, 1), acum_tp / np.maximum(acum_tp + acum_fp, 1e-9)) if len(tp) else 0.0)

    # --- classificação da imagem: classe da detecção mais confiante (acima do limiar)
    acertos = 0
    for p in IMAGENS_TESTE:
        dets = [d for d in deteccoes if d[0] == p.name and d[2] >= conf]
        prevista = max(dets, key=lambda d: d[2])[1] if dets else -1   # -1 = nada detectado
        acertos += prevista == CLASSE_REAL[p.name]

    return {'abordagem': nome, 'mAP@0.5 (teste)': float(np.mean(aps)), 'AP gado': aps[0], 'AP maçã': aps[1],
            'precisão@0.25': tp_total / max(tp_total + fp_total, 1), 'recall@0.25': tp_total / n_reais,
            'acurácia (imagem)': acertos / len(IMAGENS_TESTE), 'inferência (ms/img)': float(np.median(tempos))}

aval_detectores = pd.DataFrame(
    [avaliar_detector(f'A · YOLOv5 customizado ({nome})', PESOS[nome]) for nome in EXPERIMENTOS] +
    [avaliar_detector('B · YOLO tradicional (COCO)', YOLO_DIR / 'yolov5s.pt', COCO_PARA_NOSSO)])
display(aval_detectores.round(3))
"""))
    c.append(md("""
**O que esperar e o que observar.** O YOLO tradicional já "conhece" vaca e maçã,
então a pergunta não é se ele funciona, e sim **quanto o ajuste fino acrescenta**.
Duas diferenças pesam: o COCO rotula *cow* como um animal visto de forma ampla,
enquanto nossa base tem enquadramentos fechados (só a cabeça, o animal deitado);
e o COCO tem muitas maçãs pequenas em fruteiras, enquanto as nossas ocupam boa parte
da foto.
"""))

    # ------------------------------------------------------------ 8. CNN
    c.append(md(f"""
## 8. CNN treinada do zero

Uma rede convolucional pequena, sem nenhum peso pré-treinado, que recebe a
imagem inteira e responde **gado** ou **maçã**. Ela usa exatamente as mesmas
divisões de treino, validação e teste.

Arquitetura (capítulo 9): três blocos **convolução + ReLU + max pooling** extraem
características; *Global Average Pooling* e *Dropout* reduzem parâmetros e
overfitting; uma saída **sigmoide** dá a probabilidade de "maçã". Com só 64
imagens de treino, usamos **aumento de dados** (espelhamento, rotação, zoom e
contraste), *early stopping* na perda de validação e redução da taxa de
aprendizado quando a perda estabiliza.

**Como escolhemos a configuração — sem olhar o teste.** A primeira versão (160 px,
com *Batch Normalization* e paciência 15) ficava presa em ~50% de acerto: com
lotes de 16 imagens e só 64 exemplos, as estatísticas da normalização oscilam e o
*early stopping* devolvia os pesos das primeiras épocas. Comparamos variações
**usando apenas treino e validação**, com duas sementes aleatórias:

| Configuração | Validação (semente 42) | Validação (semente 7) |
|---|---|---|
| 160 px, BatchNorm, paciência 15 | 62% — não aprendeu | 50% — não aprendeu |
| 128 px, BatchNorm, paciência 40 | 100% | 50% — não aprendeu |
| **128 px, sem BatchNorm, paciência 40** | **100%** | **100%** |

Ficamos com a única configuração estável nas duas sementes. O teste só é usado
depois, uma vez, para medir o resultado.
"""))
    c.append(py(f"""
import tensorflow as tf                                     # Keras (TensorFlow) para a CNN
from tensorflow import keras
from tensorflow.keras import layers

SEMENTE = 42
random.seed(SEMENTE); np.random.seed(SEMENTE); tf.random.set_seed(SEMENTE)   # reprodutibilidade
LADO = 128                                                  # imagens redimensionadas para 128x128

def carregar(etapa):
    \"\"\"Lê as imagens de uma etapa e devolve (X, y), com y = 0 para gado e 1 para maçã.\"\"\"
    caminhos = sorted((DATASET / 'images' / etapa).glob('*.jpg'))
    X = np.stack([np.asarray(Image.open(p).convert('RGB').resize((LADO, LADO))) for p in caminhos]).astype('float32')
    y = np.array([0 if p.name.startswith('gado') else 1 for p in caminhos])
    return X, y, [p.name for p in caminhos]

X_tr, y_tr, _ = carregar('train'); X_va, y_va, _ = carregar('val'); X_te, y_te, nomes_te = carregar('test')
print('treino', X_tr.shape, '| validação', X_va.shape, '| teste', X_te.shape)

aumento = keras.Sequential([layers.RandomFlip('horizontal'), layers.RandomRotation(0.08),
                            layers.RandomZoom(0.15), layers.RandomContrast(0.2)], name='aumento')

def bloco(x, filtros):
    \"\"\"Convolução 3x3 + ReLU + max pooling 2x2 (sem BatchNorm: instável com só 64 imagens).\"\"\"
    x = layers.Conv2D(filtros, 3, padding='same', activation='relu')(x)
    return layers.MaxPooling2D()(x)

entrada = keras.Input((LADO, LADO, 3))
x = aumento(entrada)                                        # só atua no treino
x = layers.Rescaling(1 / 255)(x)                            # pixels 0-255 → 0-1
for filtros in (32, 64, 128):
    x = bloco(x, filtros)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
saida = layers.Dense(1, activation='sigmoid')(x)            # probabilidade de ser maçã
cnn = keras.Model(entrada, saida, name='cnn_do_zero')
cnn.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
cnn.summary()
"""))
    c.append(py(f"""
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True),  # para quando a validação não melhora
             keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-5)]   # passos menores no platô
t0 = time.time()
hist_cnn = cnn.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs={ep_cnn}, batch_size=16,
                   callbacks=callbacks, verbose=0)
tempo_cnn = time.time() - t0
print(f'Treino: {{len(hist_cnn.history["loss"])}} épocas em {{tempo_cnn:.1f}} s '
      f'(melhor época pela validação: {{int(np.argmin(hist_cnn.history["val_loss"])) + 1}})')

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
for chave, rot in (('loss', 'treino'), ('val_loss', 'validação')):
    a1.plot(hist_cnn.history[chave], label=rot)
for chave, rot in (('accuracy', 'treino'), ('val_accuracy', 'validação')):
    a2.plot(hist_cnn.history[chave], label=rot)
a1.set_title('CNN do zero — perda'); a2.set_title('CNN do zero — acurácia')
for ax in (a1, a2): ax.set_xlabel('época'); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(RESULTADOS / 'cnn_curvas.png', dpi=120); mostrar(fig)
"""))
    c.append(py("""
prever = lambda X: cnn(X, training=False).numpy().ravel()  # chamada direta: sem o overhead do predict() por lote
prever(X_te[:1])                                            # aquecimento
tempos_cnn = []
for i in range(len(X_te)):                                  # uma imagem por vez, como numa câmera
    t0 = time.perf_counter(); prever(X_te[i:i + 1]); tempos_cnn.append((time.perf_counter() - t0) * 1000)
prob = prever(X_te)                                         # probabilidade de ser maçã
pred = (prob >= 0.5).astype(int)

acuracia_cnn = float((pred == y_te).mean())
acuracia_cnn_val = float(((prever(X_va) >= 0.5).astype(int) == y_va).mean())
print(f'Acurácia — validação: {acuracia_cnn_val:.0%} | teste: {acuracia_cnn:.0%}')
display(pd.DataFrame({'imagem': nomes_te, 'real': [NOMES[v] for v in y_te],
                      'prevista': [NOMES[v] for v in pred], 'P(maçã)': prob.round(3)}))
"""))

    # ------------------------------------------------------- 9. comparação final
    c.append(md("""
## 9. Comparação final das três abordagens

Os quatro critérios pedidos no enunciado: **facilidade de uso/integração**,
**precisão**, **tempo de treinamento/customização** e **tempo de inferência**.
"""))
    c.append(py("""
treino_yolo = {f'A · YOLOv5 customizado ({n})': tempos_treino[n] for n in EXPERIMENTOS}
final = aval_detectores.set_index('abordagem')[['mAP@0.5 (teste)', 'acurácia (imagem)', 'inferência (ms/img)']].copy()
final['tempo de treino (s)'] = [treino_yolo.get(a, 0.0) for a in final.index]
final.loc['C · CNN do zero (classificação)'] = [np.nan, acuracia_cnn, float(np.median(tempos_cnn)), tempo_cnn]
final['localiza o objeto?'] = ['sim', 'sim', 'sim', 'não']
final['precisa treinar?'] = ['sim (ajuste fino)', 'sim (ajuste fino)', 'não', 'sim (do zero)']
display(final.round(3))
final.to_csv(RESULTADOS / 'comparacao_final.csv')

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 4.8))
rotulos = [i.split(' · ')[0] + '\\n' + i.split(' · ')[1].replace(' (', '\\n(') for i in final.index]
cores = ['#e4572e', '#c0392b', '#7f8c8d', '#2e86ab']
a1.bar(rotulos, final['acurácia (imagem)'], color=cores); a1.set_ylim(0, 1.05); a1.set_title('Acurácia de classificação (teste)')
a2.bar(rotulos, final['tempo de treino (s)'] / 60, color=cores); a2.set_title('Tempo de treino (min)')
a3.bar(rotulos, final['inferência (ms/img)'], color=cores); a3.set_title('Tempo de inferência (ms por imagem)')
for ax in (a1, a2, a3): ax.tick_params(axis='x', labelsize=8); ax.grid(axis='y', alpha=.3)
plt.tight_layout(); plt.savefig(RESULTADOS / 'comparacao_final.png', dpi=120); mostrar(fig)
"""))
    c.append(md("### Avaliação crítica\n\n" + AVALIACAO_CRITICA))

    # ------------------------------------------------------- 10. salvar
    c.append(md("""
## 10. Resultados salvos no Google Drive

Pesos treinados, curvas, tabelas e as imagens processadas ficam em
`MyDrive/FarmTech_Fase6/resultados/`, para consulta sem precisar re-executar.
"""))
    c.append(py("""
for nome, pesos in PESOS.items():                           # pesos best.pt de cada simulação
    shutil.copy(pesos, RESULTADOS / f"best_{nome.split()[0]}ep.pt")
for pasta in list(pastas_detect.values()) + [pasta_coco]:   # imagens de teste processadas (runs/detect/expX)
    shutil.copytree(pasta, RESULTADOS / 'detect' / pasta.name, dirs_exist_ok=True)
for nome in EXPERIMENTOS:                                   # curvas e métricas completas de cada treino
    shutil.copytree(PESOS[nome].parents[1], RESULTADOS / 'train' / PESOS[nome].parents[1].name,
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.pt'))
cnn.save(RESULTADOS / 'cnn_do_zero.keras')
print('\\n'.join(sorted(str(p.relative_to(PROJETO)) for p in RESULTADOS.rglob('*') if p.is_file())[:40]))
"""))

    # ------------------------------------------------------- 11. fechamento
    c.append(md("""
## 11. Pontos fortes, limitações e próximos passos
""" + FECHAMENTO + """
---
**Referências**

- FIAP — Fase 6, capítulos 3 (ESP32 e visão computacional), 9 (CNN) e 10 (detecção e segmentação).
- Ultralytics. *YOLOv5*. <https://github.com/ultralytics/yolov5>
- Kuznetsova, A. et al. *The Open Images Dataset V4*. IJCV, 2020. Imagens CC BY 2.0 — autoria em `dataset/fontes.csv`.
- Skalski, P. *Make Sense*. <https://www.makesense.ai>
- Lin, T.-Y. et al. *Microsoft COCO: Common Objects in Context*. ECCV, 2014.
"""))
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="versão curta para teste local")
    args = ap.parse_args()
    nb = nbf.v4.new_notebook()
    nb.cells = celulas(args.rapido)
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    }
    destino = AQUI / (("rapido_" + NOME) if args.rapido else NOME)
    nbf.write(nb, destino)
    print(f"{destino.name}: {len(nb.cells)} células")


if __name__ == "__main__":
    main()
