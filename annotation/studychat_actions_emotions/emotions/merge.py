import json
from collections import Counter
# ====== Réglages ======
FICHIERS = {
    "gemma":   "results_gemma_studychat.json",
    "qwen":    "results_qwen_studychat.json",
    "mistral": "results_mistral_studychat.json",
}
# Ordre de fiabilité (du plus fiable au moins fiable) pour départager les égalités.
FIABILITE = ["qwen", "mistral", "gemma"]

# ====== Chargement des 3 fichiers ======
def charger(chemin):
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)
resultats = {nom: charger(chemin) for nom, chemin in FICHIERS.items()}

# ====== Vote pour un turn donné ======
def voter(labels):
    labels = {m: l for m, l in labels.items() if l}
    if not labels:
        return None
    comptes = Counter(labels.values())
    max_votes = max(comptes.values())
    gagnants = [emo for emo, c in comptes.items() if c == max_votes]
    if len(gagnants) == 1:
        return gagnants[0]
    for modele in FIABILITE:
        if modele in labels and labels[modele] in gagnants:
            return labels[modele]
    return None

import unicodedata
def normaliser(label):
    if not label:
        return label
    label = label.lower().strip()
    label = unicodedata.normalize('NFKD', label)
    label = ''.join(c for c in label if not unicodedata.combining(c))
    return label

# ====== Fusion + vote sur tous les turns ======
ref = resultats["gemma"]
final = {}
for student_id, turns in ref.items():
    final[student_id] = {}
    for turn_id in turns:
        labels = {}
        for modele, data in resultats.items():
            turn_data = data.get(student_id, {}).get(turn_id)
            if turn_data:
                labels[modele] = normaliser(turn_data.get("emotion"))
        final[student_id][turn_id] = {
            "emotion_vote": voter(labels),
            "votes": labels,
        }

# ====== Sauvegarde ======
with open("results_vote_studychat.json", "w", encoding="utf-8") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)
print("Terminé ! Résultat dans results_vote_studychat.json")
