SYSTEM = (
    "Tu es un annotateur d'émotions dans des conversations étudiant-LLM. "
    "Tu réponds UNIQUEMENT avec du JSON valide, rien d'autre."
)


def build_prompt(conversation_so_far, turn_id):
    return f"""Tu annotes les états cognitivo-affectifs d'étudiants échangeant avec un LLM pendant un hackathon. Les messages contiennent souvent du code, des traces d'erreur et des requêtes techniques.

Cadre : modèle du déséquilibre cognitif (D'Mello & Graesser). Un étudiant qui progresse rencontre un obstacle → confusion. S'il le résout, il retourne à l'engagement. Sinon, la confusion se mue en frustration, puis en boredom.

Toutes les classes s'interprètent RELATIVEMENT au contexte : lis la trajectoire de l'étudiant à travers les tours, pas le message isolé.
Les réponses du LLM sont là pour le contexte : tu annotes l'état de L'ÉTUDIANT, jamais celui du LLM.

Définitions :
- engagement : la tâche progresse ; le message s'appuie sur le tour précédent, valide un résultat, ou enchaîne sur l'objectif suivant
- confusion : première rencontre avec une incompréhension — une erreur, un comportement du code, ou une réponse du LLM qui ne fait pas sens
- frustration : l'incompréhension PERSISTE à travers plusieurs tours ; tentatives répétées, reformulations, blocage qui dure
- boredom : désengagement de la tâche ; l'étudiant abandonne l'effort, délègue passivement, ou exprime le désintérêt
- curiosite : exploration au-delà du besoin immédiat, alors que rien ne bloque
- neutral : requête transactionnelle qui n'entretient aucun rapport affectif avec ce qui précède

Règles :
- Évalue l'état de L'ÉTUDIANT, pas le contenu du code. Une trace d'erreur n'est pas automatiquement de la frustration.
- Ce qui sépare confusion de frustration est la RÉPÉTITION dans les tours précédents, pas le ton du message seul.
- Ce qui sépare engagement de neutral est la CONTINUITÉ avec les tours précédents, pas la politesse du message.

Voici la conversation jusqu'ici :
{conversation_so_far}

Classe UNIQUEMENT le dernier message étudiant (Tour {turn_id}), en tenant compte du contexte.

Réponds avec EXACTEMENT ce JSON :
{{
  "{turn_id}": {{
    "emotion": "<engagement|confusion|frustration|boredom|curiosite|neutral>"
  }}
}}"""
