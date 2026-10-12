# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
<a href= "https://www.fiap.com.br/"><img src="assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Administração Paulista" border="0" width=40% height=40%></a>
</p>

<br>

# FarmTech Solutions — Visão Computacional com YOLO

## Grupo 51 · Fase 6 · Cap 1 — O despertar da Rede Neural

## 👨‍🎓 Integrantes:
- Douglas Felicio da Silva — RM572312
- Gabriel Kaique Portel Silva — RM572500
- Richard Wrobel dos Santos — RM573998

## 👩‍🏫 Professores:
### Tutor(a)
- Nicolly Candida Rodrigues de Souza
### Coordenador(a)
- André Godoi Chiovato

---

## 📜 Descrição

A FarmTech Solutions passou a atender saúde animal e segurança de propriedades
rurais, e um cliente quer ver na prática como funciona um sistema de visão
computacional. Construímos um detector **YOLOv5** que reconhece dois objetos bem
diferentes, ambos ligados ao campo — **gado bovino** e **maçã** — e o comparamos
com duas alternativas: a **YOLO tradicional** (pesos do COCO, sem treino) e uma
**CNN treinada do zero**.

| | Entrega | O que foi feito |
|---|---|---|
| **1** | YOLO customizado | base de 80 imagens (40 + 40), rotulada no Make Sense, organizada no Google Drive; treino no Colab com **30** e **60** épocas; validação, teste e prints |
| **2** | Comparação | YOLO customizado × YOLO tradicional × CNN do zero, nos quatro critérios do enunciado |

### ▶️ O relatório completo está no notebook

| | Link |
|---|---|
| **Notebook (GitHub)** | [`notebook/DouglasFelicioDaSilva_rm572312_pbl_fase6.ipynb`](notebook/DouglasFelicioDaSilva_rm572312_pbl_fase6.ipynb) |
| **Abrir no Colab** | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DougFel/fiap-fase6-grupo51-farmtech-yolo/blob/main/notebook/DouglasFelicioDaSilva_rm572312_pbl_fase6.ipynb) |
| **Vídeo de demonstração** | <!-- VIDEO --> |

O notebook traz o passo a passo das três etapas (treino, validação e teste), os
gráficos, as tabelas e a análise crítica. Este README só apresenta o projeto e
aponta o caminho.

<!-- RESULTADOS -->

---

## 📁 Estrutura de pastas

```
.
├── README.md
├── notebook/
│   ├── DouglasFelicioDaSilva_rm572312_pbl_fase6.ipynb   relatório completo (executado no Colab)
│   └── gerar_notebook.py                                fonte do notebook (texto e código versionados)
├── dataset/
│   ├── images/{train,val,test}/     80 imagens: 32/4/4 de cada classe
│   ├── labels/{train,val,test}/     rótulos YOLO desenhados no Make Sense
│   ├── makesense/                   pacote exportado do Make Sense + lista de rótulos
│   └── fontes.csv                   autoria e licença (CC BY 2.0) de cada imagem
├── scripts/
│   ├── montar_dataset.py            reconstrói a base a partir do Open Images V7
│   └── selecao_curada.csv           as 80 imagens escolhidas na curadoria visual
├── assets/                          logo e figuras deste README
└── video/                           roteiro e produção do vídeo
```

## 🔧 Como executar o código

**No Google Colab (recomendado).**

1. Clique em **Abrir no Colab** (tabela acima).
2. `Ambiente de execução → Alterar o tipo de ambiente de execução → GPU T4`.
3. `Ambiente de execução → Executar tudo` e autorize o acesso ao Google Drive.

Na primeira execução o notebook copia a base (imagens e rótulos) para
`MyDrive/FarmTech_Fase6/dataset/` e grava tudo o que produz em
`MyDrive/FarmTech_Fase6/resultados/`. Com GPU T4 leva cerca de 15 minutos.

**Versões.** Python 3, PyTorch e TensorFlow da imagem padrão do Colab; YOLOv5
fixado no commit `f957879` da Ultralytics (o notebook faz o checkout).

**Reconstruir a base (opcional).** `python scripts/montar_dataset.py` baixa as
mesmas 80 imagens do Open Images V7 a partir de `scripts/selecao_curada.csv`.

## 🗃 Histórico de lançamentos

* 1.0.0 - 13/10/2026
    * Entregas 1 e 2: notebook executado no Colab, README e vídeo
* 0.1.0 - 12/10/2026
    * Base de imagens curada, rotulada no Make Sense e organizada para o YOLO

## 📋 Licença

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/agodoi/template">MODELO GIT FIAP</a> por <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://fiap.com.br">Fiap</a> está licenciado sobre <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution 4.0 International</a>.</p>

As imagens da base são do Open Images V7, licença CC BY 2.0; a autoria de cada
uma está em `dataset/fontes.csv`.
