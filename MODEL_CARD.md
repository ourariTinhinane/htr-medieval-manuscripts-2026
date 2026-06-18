# Model Card — TrOCR Medieval French HTR 2026

## Description

Fine-tuning de `microsoft/trocr-base-handwritten` sur un corpus de manuscrits
médiévaux en ancien et moyen français (XIIIe – XVe siècle).

## Performances

|Métrique|Baseline (sans FT)|Ce modèle|
|-|-|-|
|CER|13.77%|**5.62%**|
|WER|42.41%|**15.46%**|
|Needs review|16.2%|**3.6%**|

IC Bootstrap 95% sur le CER : \[4.83%, 6.48%] — N = 1 000 — 5 419 lignes testées

## Données d'entraînement

|Corpus|Lignes|Période|
|-|-|-|
|CREMMA Medieval|45 696|XIIIe-XVe s.|
|GalliCorpora XVe|2 070|XVe s.|
|Fabliaux|11 874|XIIIe-XIVe s.|
|**Total**|**59 640**||

Split : 80% train / 10% val / 10% test — par fichier XML

## Hyperparamètres

* Base model : microsoft/trocr-base-handwritten
* Epochs : 5
* Batch size : 4
* Learning rate : 5e-5
* Warmup steps : 500
* Seed : 42

## Limitations

* Entraîné principalement sur XIVe-XVe siècle — moins précis sur XIIe-XIIIe
* Convention semi-diplomatique héritée des corpus (abréviations développées)
* Fine-tuning complet (pas LoRA) — amélioration prévue
* Performances dégradées sur pages très dégradées ou enluminées

## Usage

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from\_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from\_pretrained(
    "votre-groupe/trocr-medieval-french-2026",
    local\_files\_only=True
)

image = Image.open("ligne\_manuscrit.jpg").convert("RGB")
pixel\_values = processor(image, return\_tensors="pt").pixel\_values
generated\_ids = model.generate(pixel\_values, max\_new\_tokens=128)
texte = processor.batch\_decode(generated\_ids, skip\_special\_tokens=True)\[0]
```

## Citation

```
@misc{htr-medieval-2026,
  title={HTR Medieval French Manuscripts 2026},
  author={\[Votre groupe]},
  year={2026},
  institution={HETIC Master Data/IA},
}
```

