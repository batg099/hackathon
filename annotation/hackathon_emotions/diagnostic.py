import json
from collections import Counter
from itertools import combinations

with open("results_vote.json", encoding="utf-8") as f:
    final = json.load(f)

MODELES = ["gemma", "qwen", "mistral"]
CHAMP = "votes"  # dict {modele: label}

# --- 1. Profil d'accord ---
profil = Counter()
for turns in final.values():
    for t in turns.values():
        labels = [v for v in t[CHAMP].values() if v]
        if len(labels) < 3:
            profil["incomplet"] += 1
            continue
        n = len(set(labels))
        profil[{1: "3/3 unanime", 2: "2/3 majorite", 3: "0 accord"}[n]] += 1

total = sum(profil.values())
print("Profil d'accord (emotions) :")
for k, n in profil.most_common():
    print(f"  {k:15s} {n:5d}  ({100*n/total:.1f}%)")

# --- 2. Paires en desaccord ---
confusions = Counter()
for turns in final.values():
    for t in turns.values():
        labels = [v for v in t[CHAMP].values() if v]
        for a, b in combinations(sorted(set(labels)), 2):
            confusions[(a, b)] += 1

print("\nConfusions entre modeles :")
for (a, b), n in confusions.most_common(15):
    print(f"  {a:15s} <-> {b:15s} {n:5d}")

# --- 3. Accord par paire de modeles ---
print("\nAccord par paire de modeles :")
for m1, m2 in combinations(MODELES, 2):
    acc = tot = 0
    for turns in final.values():
        for t in turns.values():
            v1, v2 = t[CHAMP].get(m1), t[CHAMP].get(m2)
            if v1 and v2:
                tot += 1
                acc += (v1 == v2)
    print(f"  {m1:8s} vs {m2:8s}  {100*acc/tot:.1f}%  ({acc}/{tot})")

# --- 4. Distribution par modele (detecte les biais) ---
print("\nDistribution par modele :")
for m in MODELES:
    dist = Counter()
    for turns in final.values():
        for t in turns.values():
            v = t[CHAMP].get(m)
            if v:
                dist[v] += 1
    tot_m = sum(dist.values())
    print(f"\n  {m} :")
    for emo, n in dist.most_common():
        print(f"    {emo:15s} {n:5d}  ({100*n/tot_m:.1f}%)")
