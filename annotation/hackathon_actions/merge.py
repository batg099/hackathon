import json
import unicodedata
from collections import Counter

# ====== Réglages ======
# Tes 3 fichiers de résultats (actions / dialogue acts).
FICHIERS = {
    "gemma":   "actions_gemma.json",
    "qwen":    "actions_qwen.json",
    "mistral": "actions_mistral.json",
}

# Ordre de fiabilité (du plus fiable au moins fiable).
# Sert à départager en cas d'égalité (3 labels différents).
FIABILITE = ["qwen", "mistral", "gemma"]

# Le champ à lire dans chaque turn : "specific" (label précis) ou "broad" (catégorie large).
# Voter sur "broad" (8 classes) donne un accord bien plus élevé que "specific" (31 classes).
CHAMP = "specific"

# Table specific -> broad, pour retrouver la catégorie large du label voté.
LABELS = {
    "write_code": "writing", "write_english": "writing", "conversion": "writing",
    "summarize": "writing", "writing_other": "writing",
    "edit_code": "editing", "edit_english": "editing", "editing_other": "editing",
    "python_library": "conceptual_questions",
    "programming_language": "conceptual_questions",
    "computer_science": "conceptual_questions",
    "programming_tools": "conceptual_questions",
    "mathematics": "conceptual_questions",
    "other_concept": "conceptual_questions",
    "assignment_clarification": "contextual_questions",
    "code_explanation": "contextual_questions",
    "interpret_output": "contextual_questions",
    "contextual_other": "contextual_questions",
    "assignment_information": "provide_context",
    "error_message": "provide_context",
    "code": "provide_context",
    "context_other": "provide_context",
    "verify_code": "verification", "verify_report": "verification",
    "verify_output": "verification", "verify_other": "verification",
    "chit_chat": "off_topic", "greeting": "off_topic",
    "gratitude": "off_topic", "offtopic_other": "off_topic",
    "misc_other": "misc",
}


# ====== Chargement des 3 fichiers ======
def charger(chemin):
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


resultats = {nom: charger(chemin) for nom, chemin in FICHIERS.items()}


def normaliser(label):
    if not label:
        return label
    label = label.lower().strip()
    label = unicodedata.normalize('NFKD', label)
    label = ''.join(c for c in label if not unicodedata.combining(c))
    return label


# ====== Vote pour un turn donné ======
def voter(labels):
    """
    labels = {"gemma": "write_code", "qwen": "edit_code", "mistral": "write_code"}
    Renvoie le label le plus voté (départage par fiabilité si égalité).
    """
    labels = {m: l for m, l in labels.items() if l}
    if not labels:
        return None
    comptes = Counter(labels.values())
    max_votes = max(comptes.values())
    gagnants = [lab for lab, c in comptes.items() if c == max_votes]
    if len(gagnants) == 1:
        return gagnants[0]
    # Égalité -> label du modèle le plus fiable parmi les gagnants.
    for modele in FIABILITE:
        if modele in labels and labels[modele] in gagnants:
            return labels[modele]
    return None


# ====== Fusion + vote sur tous les turns ======
# On parcourt l'UNION des (student_id, turn_id) présents dans les 3 fichiers,
# pour ne pas perdre un turn qu'un modèle aurait annoté et pas un autre.
all_ids = set()
for data in resultats.values():
    all_ids.update(data.keys())

final = {}
for student_id in all_ids:
    # union des turns pour cet étudiant, tous modèles confondus
    turn_ids = set()
    for data in resultats.values():
        turn_ids.update(data.get(student_id, {}).keys())

    final[student_id] = {}
    for turn_id in turn_ids:
        labels = {}
        for modele, data in resultats.items():
            turn_data = data.get(student_id, {}).get(turn_id)
            if turn_data:
                labels[modele] = normaliser(turn_data.get(CHAMP))

        vote = voter(labels)
        final[student_id][turn_id] = {
            "action_vote": vote,
            "broad_vote": LABELS.get(vote) if vote else None,
            "votes": labels,
        }

# ====== Sauvegarde ======
with open("actions_vote.json", "w", encoding="utf-8") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)

print("Terminé ! Résultat dans actions_vote.json")
