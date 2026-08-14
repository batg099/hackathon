import json
import unicodedata
from collections import Counter

# ====== Réglages ======
FICHIERS = {
    "gemma":   "results_gemma_actions_studychat.json",
    "qwen":    "results_qwen_actions_studychat.json",
    "mistral": "results_mistral_actions_studychat.json",
}

FIABILITE = ["qwen", "mistral", "gemma"]

CATEGORIES = {"implement", "debug", "explain", "other"}

ACTIONS = {
    "implement_detailed", "implement_abstract", "implement_assignment",
    "debug_code", "debug_error", "debug_test",
    "explain_how", "explain_concept", "explain_code",
    "explain_assignment", "explain_advice",
    "other",
}

ALIAS = {
    "implementdetailed": "implement_detailed",
    "implementabstract": "implement_abstract",
    "implementassignment": "implement_assignment",
    "debugcode": "debug_code",
    "debugerror": "debug_error",
    "debugtest": "debug_test",
    "explainhow": "explain_how",
    "explainconcept": "explain_concept",
    "explaincode": "explain_code",
    "explainassignment": "explain_assignment",
    "explainadvice": "explain_advice",
}


def charger(chemin):
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


resultats = {nom: charger(chemin) for nom, chemin in FICHIERS.items()}


# ====== Normalisation ======
def normaliser(label, valides):
    if not label:
        return None
    label = str(label).lower().strip()
    label = unicodedata.normalize("NFKD", label)
    label = "".join(c for c in label if not unicodedata.combining(c))
    label = label.replace("-", "_").replace(" ", "_")
    label = ALIAS.get(label.replace("_", ""), label)
    return label if label in valides else None


def normaliser_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.lower().strip()
        if s in ("true", "yes", "oui", "1"):
            return True
        if s in ("false", "no", "non", "0"):
            return False
    return None


def coherent(cat, action):
    if not cat or not action:
        return False
    if cat == "other":
        return action == "other"
    return action.startswith(cat + "_")


# ====== Vote ======
def voter(labels):
    labels = {m: l for m, l in labels.items() if l is not None}
    if not labels:
        return None
    comptes = Counter(labels.values())
    max_votes = max(comptes.values())
    gagnants = [x for x, c in comptes.items() if c == max_votes]
    if len(gagnants) == 1:
        return gagnants[0]
    for modele in FIABILITE:
        if modele in labels and labels[modele] in gagnants:
            return labels[modele]
    return None


# ====== Fusion ======
student_ids = set()
for data in resultats.values():
    student_ids |= set(data.keys())

final = {}
stats_action = Counter()
stats_cat = Counter()
incoherents = Counter()
invalides = Counter()

for student_id in sorted(student_ids):
    turn_ids = set()
    for data in resultats.values():
        turn_ids |= set(data.get(student_id, {}).keys())

    final[student_id] = {}
    for turn_id in sorted(turn_ids, key=lambda x: int(x) if str(x).isdigit() else x):
        v_cat, v_act, v_fup = {}, {}, {}

        for modele, data in resultats.items():
            td = data.get(student_id, {}).get(turn_id)
            if not td:
                continue

            cat = normaliser(td.get("category"), CATEGORIES)
            act = normaliser(td.get("action"), ACTIONS)
            fup = normaliser_bool(td.get("is_followup"))

            if td.get("action") and not act:
                invalides[f"{modele}:{td.get('action')}"] += 1

            if not coherent(cat, act):
                incoherents[modele] += 1
                continue

            v_cat[modele] = cat
            v_act[modele] = act
            v_fup[modele] = fup

        cat_vote = voter(v_cat)
        act_vote = voter(v_act)
        fup_vote = voter(v_fup)

        final[student_id][turn_id] = {
            "category_vote": cat_vote,
            "action_vote": act_vote,
            "is_followup_vote": fup_vote,
            "votes": {
                m: {
                    "category": v_cat.get(m),
                    "action": v_act.get(m),
                    "is_followup": v_fup.get(m),
                }
                for m in resultats
            },
            "unanime_action": len(set(v_act.values())) == 1 and act_vote is not None,
            "unanime_category": len(set(v_cat.values())) == 1 and cat_vote is not None,
            "n_votants": len(v_act),
        }

        if act_vote:
            stats_action[act_vote] += 1
        if cat_vote:
            stats_cat[cat_vote] += 1


# ====== Sauvegarde ======
with open("results_vote_actions_studychat.json", "w", encoding="utf-8") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)

tours = [t for turns in final.values() for t in turns.values()]
total = len(tours)

print(f"Terminé ! {total} tours dans results_vote_actions_studychat.json\n")

print("Distribution des categories :")
tc = sum(stats_cat.values())
for c, n in stats_cat.most_common():
    print(f"  {c:12s} {n:5d}  ({100*n/tc:5.1f}%)")

print("\nDistribution des actions :")
ta = sum(stats_action.values())
for a, n in stats_action.most_common():
    print(f"  {a:24s} {n:5d}  ({100*n/ta:5.1f}%)")

n_fup = sum(1 for t in tours if t["is_followup_vote"] is True)
print(f"\nRelances : {n_fup}/{total} ({100*n_fup/total:.1f}%)   [reference papier : 10.8%]")

u_act = sum(1 for t in tours if t["unanime_action"])
u_cat = sum(1 for t in tours if t["unanime_category"])
print(f"\nUnanimite action   : {u_act}/{total} ({100*u_act/total:.1f}%)")
print(f"Unanimite category : {u_cat}/{total} ({100*u_cat/total:.1f}%)")

if incoherents:
    print("\nVotes rejetes pour incoherence category/action :")
    for m, n in incoherents.most_common():
        print(f"  {m:10s} {n:5d}")

if invalides:
    print("\nLabels invalides :")
    for k, n in invalides.most_common(10):
        print(f"  {k}  x{n}")
