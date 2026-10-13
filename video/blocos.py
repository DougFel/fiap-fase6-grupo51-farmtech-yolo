"""
Fonte única dos blocos do vídeo: âncora na página (texto de um título, com
deslocamento opcional em pixels: "título|+600") e a fala correspondente.

As durações NÃO são escritas à mão: o produzir.py sintetiza cada fala, mede quanto
tempo ela ocupa e só então define a janela de tela de cada bloco.
"""

PAGINA = {1: "video/pagina_video.html"}

VIDEO_1 = [
    ("FarmTech Solutions — Visão Computacional com YOLO", """
Olá! Somos o grupo cinquenta e um da FIAP: Douglas Felicio, Gabriel Kaique e
Richard Wrobel. Esta é a entrega do capítulo um da fase seis, O despertar da
Rede Neural. A FarmTech quer mostrar a um cliente como a visão computacional
funciona na prática. Treinamos uma rede YOLO para reconhecer dois objetos do campo
bem diferentes entre si: gado bovino e maçã.
"""),
    ("O repositório no GitHub", """
Tudo está num repositório público no GitHub: o notebook do Colab, a base de
imagens, os rótulos e os scripts que reconstroem a base. São oitenta fotos do
Open Images, com licença Creative Commons: quarenta de gado e quarenta de maçã,
divididas em trinta e duas para treino, quatro para validação e quatro para
teste, em cada classe.
"""),
    ("Rotulagem no Make Sense", """
A rotulagem foi feita no Make Sense: uma caixa por objeto, cento e uma caixas no
total, exportadas no formato YOLO. As imagens e os rótulos ficam organizados no
Google Drive, em pastas de treino, validação e teste.
"""),
    ("Execução no Google Colab", """
O notebook roda no Google Colab, conectado ao Drive, com GPU Tesla T4. Se a base
ainda não estiver no Drive de quem executa, o próprio notebook copia do
repositório. Então ele roda no Colab de qualquer pessoa, sem configuração.
"""),
    ("3.1 Curvas de treino|+220", """
Partimos do YOLOv5s pré-treinado e fizemos duas simulações idênticas, mudando só
o número de épocas: trinta e sessenta. Nas curvas, as perdas de validação
continuam caindo até a época sessenta, sem sinal de overfitting. Ou seja: com
trinta épocas, a rede ainda estava aprendendo.
"""),
    ("5. 30 x 60 épocas", """
No teste, com imagens que o modelo nunca viu, sessenta épocas venceram. O mAP a
cinquenta por cento subiu de zero vírgula setenta e quatro para zero vírgula
oitenta e um, e o mAP mais exigente, de zero vírgula quarenta e quatro para zero
vírgula cinquenta e sete. A precisão foi de sessenta e um para setenta e nove por
cento. Cada treino levou cerca de dois minutos.
"""),
    ("4.1 Prints do teste|+1000", """
Estes são os prints do teste com o modelo de sessenta épocas, gravados em runs,
detect, exp dois. Ele acerta as fotos com um animal e todas as maçãs. O ponto
fraco é o rebanho: na última foto de gado, com três animais, ele não detecta
nada. É o limite da nossa base, que tem poucas fotos de grupo.
"""),
    ("7. YOLO tradicional|+215", """
Na entrega dois, comparamos com a YOLO tradicional, usando os pesos originais do
COCO, sem treino nenhum. Ela separa bem cada vaca do rebanho, porque aprendeu com
milhões de imagens. Mas não reconhece a maçã verde em close e cria caixas demais
na fruteira.
"""),
    ("8. CNN treinada do zero|+170", """
Também treinamos uma CNN do zero, que classifica a imagem inteira. A primeira
versão, com Batch Normalization, ficou presa em cinquenta por cento. Comparando
variações só na validação, sem olhar o teste, a versão sem normalização foi a
única estável, e acertou as oito imagens de teste.
"""),
    ("9. Comparação final|+130", """
Na mesma régua, o YOLO customizado chegou a zero vírgula oitenta de mAP, contra
zero vírgula sessenta e sete do modelo pronto: treze pontos de ganho com só
sessenta e quatro fotos. A CNN acerta todas as classificações com trinta segundos
de treino, mas não diz onde o objeto está. E as três rodam bem abaixo dos trinta
e três milissegundos de um vídeo a trinta quadros por segundo.
"""),
    ("Avaliação crítica", """
Nossa recomendação para o cliente: para reconhecer e localizar gado e maçã, o
YOLO customizado com sessenta épocas, que pode receber as imagens de um
ESP32-CAM por Wi-Fi. A YOLO tradicional é um bom ponto de partida para classes do
COCO, e a CNN serve para uma triagem rápida e barata.
"""),
    ("11. Pontos fortes", """
Como limitações, a base é pequena, tem poucos rebanhos, e as métricas variam
entre execuções. Os próximos passos são mais fotos de grupo, imagens da câmera do
próprio cliente e a integração com o ESP32-CAM. O relatório completo está no
notebook, com o link no README do repositório. Obrigado!
"""),
]

VIDEOS = {1: VIDEO_1}
