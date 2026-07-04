# HTR Medieval Manuscripts 2026

**Pipeline complet HTR + NLP pour manuscrits médiévaux en ancien/moyen français**

Projet réalisé dans le cadre du Master Data/IA — Module Vision par ordinateur \& NLP — HETIC 2026.

\---

## Description

Pipeline de bout en bout en deux volets :

**Volet 1 — Computer Vision \& HTR**

1. Prétraitement des images (deskewing, CLAHE, binarisation Sauvola)
2. Segmentation layout via YOLO entraîné sur CATMuS médiéval
3. Transcription HTR via TrOCR fine-tuné sur corpus CREMMA/GalliCorpora/Fabliaux
4. Export PAGE XML + JSON livrable au module NLP

**Volet 2 — NLP**

1. Ingestion \& validation du data contract JSON
2. Normalisation des abréviations médiévales (règles + table)
3. NER par gazetier (faible supervision) + CamemBERT-LoRA (en cours)
4. POS et lemmatisation via pie\_extended (modèle freem)
5. Extraction de relations par co-occurrence
6. Graphe de connaissances (NetworkX) + export TEI-XML

\---

## Corpus d'entraînement

|Corpus|Source|Période|Fichiers|Lignes|
|-|-|-|-|-|
|CREMMA Medieval|HTR-United / CC-BY|XIIIe-XVe s.|558|45 696|
|GalliCorpora XVe|Gallica / CC-BY|XVe s.|25|2 070|
|Fabliaux médiévaux|HTR-United / CC-BY|XIIIe-XIVe s.|181|11 874|
|**TOTAL**|||**764**|**59 640**|

Split : 80% train / 10% val / 10% test — séparé par fichier XML.
Hash SHA-256 du test set : voir `data/test\_set\_hash.txt`

\---

## Résultats HTR

|Métrique|Baseline (sans FT)|Notre modèle|Seuil validation|Seuil excellence|
|-|-|-|-|-|
|CER|13.77%|**5.62%** |< 15%|< 8%|
|WER|42.41%|**15.46%** |< 25%|< 15%|
|Needs review|16.2%|**3.6%** |< 30%|< 20%|

IC Bootstrap 95% sur le CER : \[4.83%, 6.48%] — N = 1 000 ré-échantillonnages — 5 419 lignes testées

\---

## Résultats NLP

|Étape|Statut|Détail|
|-|-|-|
|Normalisation abréviations|Fait|Table de règles (q̃→que, ꝯ→con/com, u/v, i/j)|
|NER gazetier|Fait|mabile, tiece, fouchier (PER) · dame, sire (TITLE)|
|Export CoNLL-2003|Fait|200 phrases annotées|
|CamemBERT-LoRA|Fait|Entraînement à finaliser|
|POS/lemmes (freem)|Fait|pie\_extended à installer|
|Extraction relations|Fait|2 triplets · relations.json|
|Graphe NetworkX|Fait|5 nœuds · 2 arêtes · knowledge\_graph.jsonld|
|TEI-XML|Fait|2 fichiers · tei\_output/|

\---

## Prétraitement — étapes appliquées

|Étape|Outil|Paramètres|Impact|
|-|-|-|-|
|Niveaux de gris|cv2.cvtColor|BGR→GRAY|Réduction dimensionnelle|
|Deskewing|cv2.HoughLines|threshold=200|Correction inclinaison|
|CLAHE|cv2.createCLAHE|clipLimit=2.0, tileGrid=(8,8)|Amélioration contraste local|
|Binarisation Sauvola|skimage.threshold\_sauvola|window\_size=25|Binarisation adaptative|

\---

## Normalisation — règles et impact

|Règle|Exemple avant|Exemple après|
|-|-|-|
|q̃ → que|q̃ les cheualiers|que les cheualiers|
|Tilde nasale|grãt, bõne|grant, bonne|
|ꝯ → con/com|ꝯme|comme|
|ꝑ → par|ꝑ dieu|par dieu|
|⁊ → et|⁊ si|et si|
|u/v|uenir, auoit|venir, avoit|
|i/j|ioie, iuger|joie, juger|

Impact chiffré : voir `nlp/normalisation\_report.json`

\---

## Installation

```bash
git clone https://github.com/ourariTinhinane/htr-medieval-manuscripts-2026.git
cd htr-medieval-manuscripts-2026
pip install -r requirements.txt
```

\---

## Reproduire les résultats

### Volet 1 — HTR

**1. Préparer les données**

```bash
# Télécharger les corpus depuis HTR-United et HuggingFace
# Placer dans data/cremma/, data/gallicorpora/, data/fabliaux/
# Ouvrir et exécuter : notebooks/projet\_HTR.ipynb (cellule 1 à 4)
```

**2. Fine-tuning TrOCR**

```bash
# Nécessite un GPU (Google Colab T4 recommandé)
# Ouvrir et exécuter : notebooks/projet\_HTR.ipynb (cellule 5 à 7)
# Seed fixé à 42 — résultats reproductibles
```

**3. Pipeline sur images Gallica (sans vérité terrain)**

```bash
# Ouvrir et exécuter : notebooks/HTR\_au\_NLP.ipynb (cellules 1 à 3)
# Images dans : data/gallica/images\_originales/
# Sorties : page\_xml/ et dataset\_nlp/transcriptions\_nouvelles\_images.json
```

### Volet 2 — NLP

**4. Normalisation + NER + Graphe**

```bash
# Ouvrir et exécuter : notebooks/HTR\_au\_NLP.ipynb (cellules 4 à fin)
# Sorties : nlp/corpus\_normalise\_taln.json
#           nlp/corpus\_ner.conll
#           nlp/relations.json
#           nlp/knowledge\_graph.jsonld
#           nlp/tei\_output/
```

**5. Tests automatisés**

```bash
pip install pytest jsonschema
pytest tests/ -v
```

**Résultats attendus :**

* CER moyen sur test set : 5.62%
* WER moyen : 15.46%
* Needs review : 3.6%

\---

## Modèles publiés

* Camembert fine-tuné (NLP) : https://huggingface.co/TinhinaneO/camembert-ner-medieval-french-2026
* TrOCR fine-tuné (CV) : https://huggingface.co/TinhinaneO/trocr-medieval-french-2026


\---

## Licences

* Corpus : CC-BY (CREMMA, GalliCorpora, Fabliaux)
* Modèles : microsoft/trocr-base-handwritten (MIT), biglam/medieval-manuscript-yolov11 (Apache 2.0)
* Code : MIT
