import json
import unicodedata
from collections import Counter

# ====== Réglages ======
# Tes 3 fichiers de résultats. Adapte les noms/chemins si besoin.
FICHIERS = {
    "gemma":   "results_gemma.json",
    "qwen":    "results_qwen.json",
    "mistral": "results_mistral.json",
}

# Ordre de fiabilité (du plus fiable au moins fiable).
# Sert à départager en cas d'égalité (3 labels différents).
FIABILITE = ["qwen", "mistral", "gemma"]   # ← mets-les dans TON ordre de confiance


# ====== Normalisation des labels ======
# On retire d'abord les accents, PUIS on mappe les variantes vers une forme
# canonique. Indispensable car les modèles produisent parfois la forme anglaise
# (curiosity) et parfois la forme française (curiosite/curiosité) : sans ce
# mapping, le vote les traite comme deux classes distinctes et perd le consensus.
ALIAS = {
    "curiosite":   "curiosity",
    "curiosity":   "curiosity",
    "ennui":       "boredom",
    "boredom":     "boredom",
    "neutre":      "neutral",
    "neutral":     "neutral",
    "engagement":  "engagement",
    "confusion":   "confusion",
    "frustration": "frustration",
}


def normaliser(label):
    if not label:
        return label
    # minuscules + suppression des accents
    label = label.lower().strip()
    label = unicodedata.normalize('NFKD', label)
    label = ''.join(c for c in label if not unicodedata.combining(c))
    # traduction / canonisation (curiosite -> curiosity, etc.)
    return ALIAS.get(label, label)


# ====== Chargement des 3 fichiers ======
def charger(chemin):
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


resultats = {nom: charger(chemin) for nom, chemin in FICHIERS.items()}


# ====== Vote pour un turn donné ======
def voter(labels):
    """
    labels = {"gemma": "engagement", "qwen": "confusion", "mistral": "engagement"}
    Renvoie l'émotion la plus votée (départage par fiabilité si égalité).
    """
    # On ignore les modèles qui n'ont pas de label pour ce turn.
    labels = {m: l for m, l in labels.items() if l}
    if not labels:
        return None
    # Compter les votes.
    comptes = Counter(labels.values())
    max_votes = max(comptes.values())
    gagnants = [emo for emo, c in comptes.items() if c == max_votes]
    # Un seul gagnant -> c'est lui.
    if len(gagnants) == 1:
        return gagnants[0]
    # Égalité -> on prend le label du modèle le plus fiable.
    for modele in FIABILITE:
        if modele in labels and labels[modele] in gagnants:
            return labels[modele]
    return None


# ====== Fusion + vote sur tous les turns ======
# On se base sur les Id/Turn présents dans le premier modèle.
# (on suppose que les 3 fichiers couvrent les mêmes turns)
ref = resultats["gemma"]   # référence pour parcourir les Id et Turn
final = {}
for student_id, turns in ref.items():
    final[student_id] = {}
    for turn_id in turns:
        # Récupérer le label "etudiant" de chaque modèle pour ce (student_id, turn_id).
        labels = {}
        for modele, data in resultats.items():
            turn_data = data.get(student_id, {}).get(turn_id)
            if turn_data:
                labels[modele] = normaliser(turn_data.get("emotion"))
        # Voter et stocker le résultat avec le détail des 3 votes.
        final[student_id][turn_id] = {
            "emotion_vote": voter(labels),
            "votes": labels,   # garde la trace de qui a voté quoi
        }


# ====== Sauvegarde ======
with open("results_vote.json", "w", encoding="utf-8") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)

print("Terminé ! Résultat dans results_vote.json")
