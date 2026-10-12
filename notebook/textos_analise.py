"""
Textos de análise do notebook, escritos a partir da execução no Colab que está
versionada (GPU Tesla T4, 12/10/2026). Os números citados são os dessa execução.
Usados pelo gerar_notebook.py e para completar o notebook executado.
"""

CONCLUSOES_ENTREGA1 = """
**1. Mais épocas ajudaram — e a rede ainda estava aprendendo aos 30.** Com 60
épocas o modelo foi melhor em todas as medidas que importam no teste:

| Teste (8 imagens, 12 objetos) | 30 épocas | 60 épocas |
|---|---|---|
| mAP@0.5 | 0,739 | **0,811** |
| mAP@0.5:0.95 (caixa precisa) | 0,439 | **0,569** |
| Precisão | 0,611 | **0,789** |
| Recall | **0,827** | 0,750 |

O ganho maior está no **mAP@0.5:0.95**, que exige caixas bem ajustadas: as
épocas extras refinaram a *localização*, não só o reconhecimento. As curvas da
seção 3.1 mostram que a perda de validação continuou caindo até a época 60 —
**não houve overfitting**, então 30 épocas interrompem um aprendizado ainda em curso.

**2. Erro.** A perda total de validação caiu de 0,016 (30 épocas) para 0,010
(60 épocas). A de treino fica em torno de 0,05 nos dois casos porque é medida com
aumento de dados (mosaico, cortes, variação de cor); por isso ela é *maior* que a
de validação, e isso não indica problema.

**3. Desempenho.** A inferência é a mesma nos dois modelos (~5 ms por imagem na
T4) — a arquitetura é idêntica, só os pesos mudam. Os tempos de treino ficaram
próximos (2,6 e 2,2 min): o primeiro treino paga custos únicos de preparação
(cache das imagens lidas do Drive, verificações iniciais), por isso o de 30 épocas
não saiu na metade do de 60. Por época, o custo real é de ~2 s.

**4. Validação × teste.** Na validação os dois modelos ficam perto do teto
(mAP@0.5 de 0,920 e 0,995), mas no teste caem para 0,74 e 0,81. A diferença não é
"defeito" do modelo e sim do tamanho das bases: a validação tem 8 imagens com
quase um objeto cada, enquanto o teste tem **duas fotos de rebanho com 3 animais**.
Os prints da seção 4.1 mostram exatamente onde o modelo erra:

- **acerta** todas as fotos com um animal e todas as maçãs;
- em `gado_test_03` agrupa os animais numa caixa só, e em `gado_test_04` (rebanho
  em fundo enevoado) **não detecta nada** — por isso o recall do gado no teste é
  0,50 enquanto o da maçã é 1,00;
- em `maca_test_04` encontra também maçãs cortadas na borda, que não foram
  rotuladas — viram falsos positivos, e é isso que puxa a precisão da maçã para baixo.

**5. Variação entre execuções.** Com 8 imagens de teste, um único objeto muda o
mAP em vários pontos, e o treino em GPU não é perfeitamente determinístico. Numa
execução anterior idêntica, o modelo de 60 épocas chegou a 0,848 de mAP@0.5 no
teste. A conclusão — **60 épocas superam 30** — se repetiu em todas as execuções;
os valores exatos devem ser lidos com essa margem.

**Mensagem para o cliente.** Com 32 fotos por classe e 2 minutos de GPU, o
detector já reconhece gado e maçã com confiança em cenas com um ou poucos
objetos. Para contar animais num rebanho, o próximo passo é treinar com mais
fotos de grupo — é a única situação em que ele falha de forma sistemática.
"""

AVALIACAO_CRITICA = """
| Critério | A · YOLOv5 customizado | B · YOLO tradicional (COCO) | C · CNN do zero |
|---|---|---|---|
| **Facilidade de uso / integração** | média: exige rotular caixas (Make Sense) e treinar; depois, um `best.pt` que roda com o mesmo `detect.py` | **a mais fácil**: zero treino, `yolov5s.pt` pronto; mas só reconhece as 80 classes do COCO | média: rótulo só por pasta (sem caixas), mas arquitetura e treino são escritos à mão |
| **Precisão (teste)** | **mAP@0.5 = 0,798** (60 ép.) — a melhor detecção; acurácia por imagem 87,5% | mAP@0.5 = 0,665 — 13 pontos abaixo; acurácia 87,5% | **acurácia 100%**, mas só diz *o que* há na foto, não *onde* |
| **Tempo de treino / customização** | ~2 min de GPU, depois de desenhar 101 caixas nas 80 fotos | nenhum | **~30 s** de GPU, rótulo trivial |
| **Tempo de inferência** | ~21 ms/imagem (com leitura e pré-processamento) | ~21 ms/imagem — mesma arquitetura | **~13 ms/imagem** — ~75 vezes menos parâmetros (93 mil × 7 milhões) |

**O que os números dizem.**

- **O ajuste fino vale a pena.** O mesmo YOLOv5s, com os mesmos pesos de partida,
  passou de 0,665 para 0,798 de mAP@0.5 só com 64 fotos rotuladas. O ganho vem da
  **maçã** (AP 0,658 → 0,850): o COCO aprendeu maçãs pequenas em fruteiras e
  **não reconheceu a maçã verde em close** de `maca_test_03`, que o modelo
  customizado acerta.
- **Mas o COCO é melhor em rebanho.** Na foto `gado_test_04`, a YOLO tradicional
  separa cada animal; a nossa não detecta nenhum. Ela viu milhares de cenas com
  várias vacas — nós, poucas. É a evidência mais clara de que **o que limita o
  modelo customizado é a base, não a técnica**.
- **A CNN acertou 100% porque o problema dela é mais fácil.** Classificar a
  imagem inteira entre "gado" e "maçã" — cor, textura e fundo já separam bem as
  duas classes — é muito mais simples que localizar cada objeto com IoU ≥ 0,5. Com
  só 8 imagens de teste, 100% significa "acertou as 8", não "nunca erra". E ela
  **não conta** objetos nem diz onde estão — não serve para contar rebanho nem
  para guiar uma câmera.
- **Tempo de inferência não é gargalo em nenhuma delas.** Todas ficam bem abaixo
  dos ~33 ms de um quadro a 30 fps — nas três, daria para processar vídeo ao vivo.

> **Observação sobre a régua.** Os mAP desta seção vêm da avaliação própria da
> seção 7.1, aplicada igualmente a A e B. Eles diferem um pouco dos da seção 4
> (`val.py` do YOLOv5) porque a interpolação da curva precisão-recall e o
> pré-processamento são diferentes; o que importa é que **A e B foram medidos com
> a mesma régua**.

**Recomendação para a FarmTech.** Para *reconhecer e localizar* gado e maçã — o
pedido do cliente —, o **YOLOv5 customizado com 60 épocas** é a escolha: melhor
mAP, mesmo custo de inferência do modelo pronto e integração direta com a câmera
(ESP32-CAM → servidor com `best.pt`, como no capítulo 3). A YOLO tradicional é um
bom *ponto de partida* para classes que já existem no COCO, e a CNN do zero é
útil quando basta saber *o que* aparece na foto, como triagem rápida e barata.
"""

FECHAMENTO = """
**Pontos fortes**

- **Pipeline completo e reprodutível:** base curada e documentada (autoria e
  licença de cada foto), rótulos do Make Sense versionados, YOLOv5 fixado num
  commit, semente fixa, e um notebook que monta o próprio Drive e baixa a base —
  roda no Colab de qualquer pessoa sem configuração.
- **Comparação justa:** as duas simulações diferem só nas épocas; os três
  modelos da Entrega 2 usam as mesmas imagens de teste e a mesma régua.
- **Avaliação honesta:** o teste nunca participou de nenhuma escolha; a
  configuração da CNN foi decidida só com a validação; os erros estão mostrados
  imagem por imagem, não escondidos na média.
- **Resultado útil:** com 64 fotos de treino, o detector supera o modelo pronto
  em 13 pontos de mAP@0.5 e já funciona em tempo real.

**Limitações**

- **Base pequena.** 8 imagens de validação e 8 de teste tornam as métricas
  instáveis: um objeto muda o mAP em vários pontos, e execuções idênticas variam
  (seção 5, item 5).
- **Poucas cenas de rebanho.** A base tem em média 1,3 objeto por imagem; o
  modelo falha justamente nas fotos com 3 animais.
- **Rotulagem incompleta em um caso.** Em `maca_test_04`, só a maçã central foi
  rotulada; as maçãs cortadas na borda contam como falsos positivos e derrubam a
  precisão da classe.
- **Fotos "de internet".** O Open Images traz fotos variadas, mas não reproduz as
  condições do cliente (câmera fixa, ângulo de cima, noite, chuva). O desempenho
  no campo precisa ser medido com imagens do próprio cliente.
- **A CNN só classifica:** não localiza nem conta objetos.

**Próximos passos**

1. Ampliar a base com fotos de rebanho e com imagens da câmera do próprio cliente,
   e rotular todas as maçãs visíveis.
2. Validação cruzada ou base de teste maior, para métricas com menos variância.
3. Testar o YOLOv5m (maior) e mais épocas, já que a curva de validação ainda subia
   aos 60.
4. Integrar com o ESP32-CAM: a câmera envia os quadros por Wi-Fi a um servidor
   que roda o `best.pt` (o "Ir Além" da fase).
"""
