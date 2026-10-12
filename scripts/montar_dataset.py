"""
Monta a base de 80 imagens do projeto a partir do Open Images V7 (Google).

Classe A: gado bovino ("Cattle", /m/01xq0k1)
Classe B: maçã        ("Apple",  /m/014j1m)

Como as imagens foram escolhidas:
  1. Filtro automático (função candidatas): foto real, sem rotação pendente,
     só uma das duas classes na imagem, até 3 objetos, nenhum objeto marcado
     como "grupo", o menor ocupando ao menos 4% da imagem e licença CC BY 2.0.
  2. Curadoria visual: a classe "Cattle" do Open Images mistura iaques, bisões,
     búfalos e até antílopes, e "Apple" inclui tortas e fatias. Olhamos todas as
     candidatas e ficamos com 40 fotos inequívocas de cada classe. A escolha está
     em scripts/selecao_curada.csv, o que torna a base reprodutível.

A divisão é a do enunciado: por classe, 32 treino, 4 validação e 4 teste
(sorteio com semente fixa). As imagens são reduzidas para o lado maior = 640 px,
o tamanho de entrada do YOLOv5, e a autoria de cada uma vai para dataset/fontes.csv.

As caixas do Open Images NÃO entram no dataset: a rotulação foi feita no Make
Sense, como pede o enunciado. As caixas de referência ficam só em _trabalho/,
usadas para conferir a rotulação.
"""
import io
import random
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
OI = RAIZ / "_trabalho" / "oi"
SAIDA = RAIZ / "dataset"
CURADORIA = RAIZ / "scripts" / "selecao_curada.csv"
CLASSES = {"/m/01xq0k1": "gado", "/m/014j1m": "maca"}
DIVISAO = {"train": 32, "val": 4, "test": 4}
SEMENTE = 42
LADO = 640


def candidatas() -> pd.DataFrame:
    """Aplica os filtros de qualidade e devolve uma linha por imagem."""
    caixas = pd.concat(
        [pd.read_csv(OI / "validation-bbox.csv"), pd.read_csv(OI / "test-bbox.csv")]
    )
    meta = pd.concat(
        [pd.read_csv(OI / "val-rot.csv"), pd.read_csv(OI / "test-rot.csv")]
    ).set_index("ImageID")

    com_classe = caixas[caixas.LabelName.isin(CLASSES)].copy()
    com_classe["area"] = (com_classe.XMax - com_classe.XMin) * (com_classe.YMax - com_classe.YMin)
    por_img = com_classe.groupby("ImageID").agg(
        rotulos=("LabelName", lambda s: set(s)),
        n=("LabelName", "size"),
        grupo=("IsGroupOf", "max"),
        desenho=("IsDepiction", "max"),
        menor=("area", "min"),
    )
    ok = (
        (por_img.rotulos.apply(len) == 1)   # só uma das duas classes
        & (por_img.grupo == 0)             # nenhum objeto "em grupo"
        & (por_img.desenho == 0)           # foto real, não ilustração
        & (por_img.n <= 3)                 # até 3 objetos
        & (por_img.menor >= 0.04)          # objetos grandes o bastante
    )
    por_img = por_img[ok & por_img.index.isin(meta.index)].copy()
    por_img["classe"] = por_img.rotulos.apply(lambda s: CLASSES[next(iter(s))])
    for col in ["Subset", "License", "Rotation", "OriginalLandingURL", "Author"]:
        por_img[col] = meta.loc[por_img.index, col]
    por_img = por_img[por_img.Rotation.fillna(0) == 0]
    por_img = por_img[por_img.License.str.contains("/by/2.0", na=False)]
    return por_img.reset_index()


def baixar(image_id: str, subset: str) -> Image.Image:
    """Baixa do espelho oficial do Open Images no S3 (as URLs do Flickr expiram)."""
    pasta = "validation" if subset == "validation" else "test"
    url = f"https://open-images-dataset.s3.amazonaws.com/{pasta}/{image_id}.jpg"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def main() -> None:
    tabela = candidatas().set_index("ImageID")
    curada = pd.read_csv(CURADORIA, comment="#", names=["classe", "ImageID"])
    assert curada.ImageID.isin(tabela.index).all(), "imagem curada fora dos filtros"

    registros = []
    for classe in ["gado", "maca"]:
        ids = curada[curada.classe == classe].ImageID.tolist()
        random.Random(SEMENTE).shuffle(ids)          # sorteio da divisão
        assert len(ids) == sum(DIVISAO.values())

        inicio = 0
        for parte, qtd in DIVISAO.items():
            pasta = SAIDA / "images" / parte
            pasta.mkdir(parents=True, exist_ok=True)
            for i, image_id in enumerate(ids[inicio:inicio + qtd], 1):
                linha = tabela.loc[image_id]
                img = baixar(image_id, linha.Subset)
                img.thumbnail((LADO, LADO))          # lado maior = 640 px
                nome = f"{classe}_{parte}_{i:02d}.jpg"
                img.save(pasta / nome, quality=92)
                registros.append({
                    "arquivo": f"images/{parte}/{nome}", "classe": classe,
                    "parte": parte, "open_images_id": image_id,
                    "autor": linha.Author, "licenca": linha.License,
                    "origem": linha.OriginalLandingURL,
                })
            inicio += qtd
        print(f"{classe}: {len(ids)} imagens gravadas")

    pd.DataFrame(registros).to_csv(SAIDA / "fontes.csv", index=False)
    print(f"fontes.csv com {len(registros)} linhas")


if __name__ == "__main__":
    main()
