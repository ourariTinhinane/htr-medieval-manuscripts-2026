# CONVENTIONS_NLP.md
## Pipeline NLP — Manuscrits médiévaux 15e siècle
### Projet HETIC MD5 2026

---

## 1. Schéma BIO

Nous utilisons le schéma **BIO (Beginning-Inside-Outside)** standard :
- `B-TYPE` : premier token d'une entité de type TYPE
- `I-TYPE` : token suivant dans la même entité (non utilisé ici car entités mono-token)
- `O` : token hors entité

**Pourquoi BIO et pas BIOES ?** Notre gazetier ne produit que des entités mono-token,
donc I- et E- ne sont jamais utilisés. BIO suffit et reste compatible avec seqeval/CoNLL-2003.

## 2. Types d'entités

| Type  | Description                        | Corpus          |
|-------|------------------------------------|-----------------|
| PER   | Personnages nommés                 | Littéraire      |
| TITLE | Titres nobiliaires/religieux       | Littéraire + Admin |
| LOC   | Lieux (villes, régions)            | Administratif   |
| DATE  | Dates et fêtes liturgiques         | Administratif   |
| ORG   | Institutions (chancellerie, etc.)  | Administratif   |

## 3. Alignement subwords (point critique)

Lors du fine-tuning CamemBERT-LoRA, les tokens BIO doivent être alignés sur les subwords.
**Stratégie first-token** : seul le premier subword d'un mot reçoit l'étiquette réelle.
Les subwords suivants reçoivent `-100` (ignorés par la CrossEntropyLoss de PyTorch).

```python
# Exemple : "mabile" → ["▁mab", "ile"]
# Labels   : [B-PER,   -100  ]
```

Référence : `align_labels_to_subwords()` dans le notebook.

## 4. Normalisation

### Règles déterministes (dans l'ordre d'application)
1. Unicode NFD (décomposition des caractères composés)
2. q̃ → que (avant la règle nasale car q n'est pas une voyelle)
3. Tilde nasale + assimilation contextuelle (m/n selon consonne suivante)
4. ꝯ → con/com (même règle d'assimilation)
5. Symboles directs : ⁊→et, ꝫ→et, ꝭ→est, ꝟ→ver
6. Symboles ambigus (flaggés) : ꝑ→par, ꝕ→pre
7. u/v contextuel (v en initiale devant voyelle, i exclu pour les diphtongues ui)
8. i/j contextuel (j en initiale devant voyelle)
9. Unicode NFC (recomposition)

### Cas non résolus (flaggés dans ambiguites[])
- Accent combinant U+0301 : sens variable selon le mot
- Tilde verticale U+033E : troncation ambiguë (fouchier/foucher/etc.)

## 5. Correction guidée par confiance

Seuil : `char_confidence < 0.70` → token candidat à correction.
Modèle : CamemBERT MLM (Masked Language Model) — `almanach/camembert-base`.
Stratégie : masquage du token incertain, évaluation des top-k candidats dans le contexte.

## 6. CER relatif

Sans vérité terrain, on mesure le **CER relatif** :
- Brut → normalisé : impact des règles de normalisation
- Normalisé → corrigé MLM : impact de la correction guidée

## 7. Modèle NER

Modèle de base : `pjox/dalembert-classical-fr-ner` (moyen français, Hugging Face).
Fallback : `almanach/camembert-base`.
Fine-tuning : LoRA r=8 sur couches query+value, 15 epochs, lr=2e-4.
Évaluation : seqeval F1 micro + F1 par type d'entité.

## 8. Reproductibilité

Seeds fixées : `random.seed(42)`, `torch.manual_seed(42)`, `np.random.seed(42)`.
Test set scellé par SHA-256 dès le début du pipeline.
Split stratifié par type de document (littéraire/administratif).
