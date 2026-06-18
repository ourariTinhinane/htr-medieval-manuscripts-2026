"""
Tests automatisés du pipeline HTR — HETIC MD5 2026.
Couvre : prétraitement, schéma JSON data contract, non-régression CER.

Exécution :
    pip install pytest jsonschema
    pytest tests/ -v
"""
import pytest
import numpy as np
import cv2
import json
import jsonschema
from pathlib import Path
from skimage.filters import threshold_sauvola


# ── Schéma du data contract ───────────────────────────────────────────────────
DATA_CONTRACT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["texte_predit", "texte_reference", "cer", "wer",
                     "needs_review", "polygone"],
        "properties": {
            "texte_predit":    {"type": "string"},
            "texte_reference": {"type": "string"},
            "cer":             {"type": "number", "minimum": 0, "maximum": 1},
            "wer":             {"type": "number", "minimum": 0},
            "needs_review":    {"type": "boolean"},
            "polygone":        {"type": "string"},
        }
    }
}


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def image_grise():
    """Image grise synthétique 200x200 pour tester le prétraitement."""
    return np.random.randint(100, 200, (200, 200), dtype=np.uint8)


@pytest.fixture
def image_couleur():
    """Image couleur synthétique 200x200."""
    return np.random.randint(100, 200, (200, 200, 3), dtype=np.uint8)


# ── Tests prétraitement ───────────────────────────────────────────────────────
class TestPretraitement:

    def test_conversion_gris_shape(self, image_couleur):
        """La conversion BGR→gris produit une image 2D."""
        gris = cv2.cvtColor(image_couleur, cv2.COLOR_BGR2GRAY)
        assert gris.ndim == 2
        assert gris.shape == (200, 200)

    def test_conversion_gris_dtype(self, image_couleur):
        """Le dtype reste uint8 après conversion."""
        gris = cv2.cvtColor(image_couleur, cv2.COLOR_BGR2GRAY)
        assert gris.dtype == np.uint8

    def test_clahe_plage_valeurs(self, image_grise):
        """CLAHE ne produit pas de valeurs hors [0, 255]."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        resultat = clahe.apply(image_grise)
        assert resultat.min() >= 0
        assert resultat.max() <= 255

    def test_clahe_shape_preservee(self, image_grise):
        """CLAHE conserve les dimensions."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        resultat = clahe.apply(image_grise)
        assert resultat.shape == image_grise.shape

    def test_sauvola_binaire(self, image_grise):
        """La binarisation Sauvola produit uniquement 0 et 255."""
        seuil = threshold_sauvola(image_grise, window_size=25)
        binaire = (image_grise > seuil).astype(np.uint8) * 255
        assert set(np.unique(binaire)).issubset({0, 255})

    def test_rotation_dimensions(self, image_grise):
        """La rotation conserve les dimensions h×w."""
        h, w = image_grise.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 1.5, 1.0)
        redresse = cv2.warpAffine(image_grise, M, (w, h))
        assert redresse.shape == (h, w)


# ── Tests schéma JSON data contract ──────────────────────────────────────────
class TestDataContract:

    def test_schema_valide(self):
        """Un exemple conforme passe la validation."""
        exemple = [{
            "texte_predit":    "Et si com il auoit",
            "texte_reference": "Et si com il auoit",
            "cer": 0.0,
            "wer": 0.0,
            "needs_review": False,
            "polygone": "100 200 300 200 300 250 100 250",
        }]
        jsonschema.validate(instance=exemple, schema=DATA_CONTRACT_SCHEMA)

    def test_schema_rejette_cer_negatif(self):
        """Un CER négatif lève une ValidationError."""
        exemple = [{
            "texte_predit": "test", "texte_reference": "test",
            "cer": -0.1, "wer": 0.0, "needs_review": False,
            "polygone": "0 0 100 0 100 50 0 50",
        }]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=exemple, schema=DATA_CONTRACT_SCHEMA)

    def test_schema_rejette_champ_manquant(self):
        """Un dict sans 'polygone' lève une ValidationError."""
        exemple = [{
            "texte_predit": "test", "texte_reference": "test",
            "cer": 0.0, "wer": 0.0, "needs_review": False,
        }]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=exemple, schema=DATA_CONTRACT_SCHEMA)


# ── Test non-régression CER ───────────────────────────────────────────────────
class TestNonRegression:

    def test_cer_sous_seuil_validation(self):
        """
        Le CER moyen sur resultats_finetuned.json doit être < 15%
        (seuil de validation du brief).
        """
        chemin = Path("resultats_finetuned.json")
        if not chemin.exists():
            pytest.skip("Fichier résultats absent (normal en CI)")
        with open(chemin, encoding="utf-8") as f:
            data = json.load(f)
        cer_moyen = np.mean([r["cer"] for r in data])
        assert cer_moyen < 0.15, (
            f"CER {cer_moyen:.4f} dépasse le seuil de validation (0.15)"
        )