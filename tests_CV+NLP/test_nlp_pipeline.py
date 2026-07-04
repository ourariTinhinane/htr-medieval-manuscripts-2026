
import pytest, json, re, unicodedata
from pathlib import Path

# ── Test 1 : Validation du schéma JSON ────────────────────────────────────────
def test_schema_json_data_contract():
    """Le JSON produit par le pipeline CV doit respecter le schéma obligatoire."""
    import jsonschema
    SCHEMA = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["image","transcriptions"],
            "properties": {
                "image": {"type":"string"},
                "transcriptions": {
                    "type":"array",
                    "items": {
                        "type":"object",
                        "required":["zone_id","bbox","texte","confidence","char_confidences","needs_review"],
                    }
                }
            }
        }
    }
    # Données de test minimales valides
    data_valide = [{
        "image": "/path/to/image.jpg",
        "nb_zones": 2,
        "transcriptions": [{
            "zone_id": 0,
            "bbox": [0,0,100,50],
            "class_name": "DefaultLine",
            "texte": "et si me sui mors a mes mains",
            "confidence": 0.95,
            "char_confidences": [0.99]*30,
            "needs_review": False,
        }]
    }]
    jsonschema.validate(data_valide, SCHEMA)  # ne doit pas lever d'exception

    # Données invalides (champ manquant)
    data_invalide = [{"image":"test.jpg"}]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data_invalide, SCHEMA)

    print("✅ Test 1 passé : schéma JSON valide")


# ── Test 2 : La normalisation ne dégrade pas le CER ───────────────────────────
def test_normalisation_ne_degrade_pas_cer():
    """
    Sur un petit échantillon de référence, la normalisation doit
    réduire ou maintenir le CER, pas l'augmenter.
    Référence humaine : paires (brut, correct) connues.
    """
    # Paires de référence : (texte brut HTR, forme correcte attendue)
    PAIRES_REFERENCE = [
        ("grãt merci",          "grant merci"),      # tilde nasale
        ("q̃ il vousist",       "que il vousist"),    # q tilde
        ("⁊ si coment",         "et si coment"),      # tironien
        ("ꝯme il dist",         "come il dist"),      # con abrégé
        ("uenir a lui",         "venir a lui"),        # u→v
        ("ioie et leece",       "joie et leece"),     # i→j
    ]

    def cer(ref, hyp):
        if not ref: return 0.0
        ref, hyp = list(ref), list(hyp)
        m, n = len(ref), len(hyp)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1): dp[i][0] = i
        for j in range(n+1): dp[0][j] = j
        for i in range(1,m+1):
            for j in range(1,n+1):
                if ref[i-1]==hyp[j-1]: dp[i][j]=dp[i-1][j-1]
                else: dp[i][j]=1+min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])
        return dp[m][n]/max(m,1)

    def _consonne_assimilee(suite):
        return 'm' if suite[:1].lower() in ('p','b','m') else 'n'

    def normaliser_simple(texte):
        t = unicodedata.normalize("NFD", texte)
        t = re.sub(r'[qQ]\u0303', 'que', t)
        t = re.sub(r'([aeiouyAEIOUY])\u0303(\w*)',
                   lambda m: m.group(1)+_consonne_assimilee(m.group(2))+m.group(2), t)
        t = re.sub(r'ꝯ(\w*)',
                   lambda m: ('com' if _consonne_assimilee(m.group(1))=='m' else 'con')+m.group(1), t)
        t = t.replace("⁊","et")
        t = unicodedata.normalize("NFC", t)
        t = re.sub(r'\buu','uv',t)
        t = re.sub(r'\bu(?=[aeouyéèê])','v',t)
        t = re.sub(r'\bi(?=[aeouyéèê])','j',t)
        return t

    for brut, correct in PAIRES_REFERENCE:
        normalise = normaliser_simple(brut)
        cer_avant = cer(correct, brut)
        cer_apres = cer(correct, normalise)
        assert cer_apres <= cer_avant + 0.05, (
            f"Normalisation dégrade le CER : {brut!r} → {normalise!r} "
            f"(attendu {correct!r}) : {cer_avant:.3f} → {cer_apres:.3f}"
        )

    print("✅ Test 2 passé : normalisation ne dégrade pas le CER")


if __name__ == "__main__":
    test_schema_json_data_contract()
    test_normalisation_ne_degrade_pas_cer()
    print("
✅ Tous les tests passés !")
