import math
import random


def distance_entre_villes(ville1, ville2):
    """Calcule la distance entre deux villes"""
    x1, y1 = ville1
    x2, y2 = ville2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

#la fonction objective a minimiser
def calculer_distance_totale(parcours, villes):
    """Calcule la distance totale d'un parcours"""
    total = 0
    n = len(villes)

    for i in range(n):
        ville_actuelle = parcours[i]
        ville_suivante = parcours[(i + 1) % n]  # Retour à la première ville
        total += distance_entre_villes(villes[ville_actuelle], villes[ville_suivante])

    return total


def creer_parcours_aleatoire(n_villes):
    """Crée un parcours aléatoire"""
    parcours = list(range(n_villes))
    random.shuffle(parcours)
    return parcours


def modifier_parcours(parcours):
    """Crée une petite modification du parcours"""
    nouveau_parcours = parcours.copy()
    i, j = random.sample(range(len(parcours)), 2)
    nouveau_parcours[i], nouveau_parcours[j] = nouveau_parcours[j], nouveau_parcours[i]
    return nouveau_parcours


def accepter_nouveau_parcours(distance_actuelle, nouvelle_distance, niveau_exploration):
    """Décide si on acecepte le nouveau parcours"""
    # Toujours accepter si c'est mieux
    if nouvelle_distance < distance_actuelle:
        return True

    # Parfois accepter même si c'est moins bon
    difference = nouvelle_distance - distance_actuelle
    probabilite = math.exp(-difference / niveau_exploration)
    return random.random() < probabilite


def mettre_a_jour_meilleur_parcours(parcours, distance, meilleur_parcours, meilleure_distance):
    """Met à jour le meilleur parcours si nécessaire"""
    if distance < meilleure_distance:
        return parcours.copy(), distance, True  # True = nouveau meilleur trouvé
    return meilleur_parcours, meilleure_distance, False


def explorer_nouveaux_parcours(parcours_actuel, distance_actuelle, meilleur_parcours,
                               meilleure_distance, villes, niveau_exploration, essais_par_niveau):
    """Explore de nouveaux parcours à un niveau d'exploration donné"""
    for _ in range(essais_par_niveau):
        # Créer un nouveau parcours modifié
        nouveau_parcours = modifier_parcours(parcours_actuel)
        nouvelle_distance = calculer_distance_totale(nouveau_parcours, villes)

        # Décider si on garde ce parcours
        if accepter_nouveau_parcours(distance_actuelle, nouvelle_distance, niveau_exploration):
            parcours_actuel = nouveau_parcours
            distance_actuelle = nouvelle_distance

            # Vérifier si c'est le nouveau meilleur
            meilleur_parcours, meilleure_distance, est_meilleur = mettre_a_jour_meilleur_parcours(
                nouveau_parcours, nouvelle_distance, meilleur_parcours, meilleure_distance
            )

            if est_meilleur:
                print(f"Nouveau meilleur: {meilleure_distance:.2f} km")

    return parcours_actuel, distance_actuelle, meilleur_parcours, meilleure_distance


def trouver_meilleur_parcours(villes, essais_par_niveau=100, refroidissement=0.95):
    """
    Trouve le meilleur parcours pour visiter toutes les villes
    en utilisant une méthode inspirée du refroidissement progressif.
    """

    # 1. Initialisation
    n_villes = len(villes)
    parcours_actuel = creer_parcours_aleatoire(n_villes)
    distance_actuelle = calculer_distance_totale(parcours_actuel, villes)

    meilleur_parcours = parcours_actuel.copy()
    meilleure_distance = distance_actuelle

    niveau_exploration = 1000

    print(f"Distance de départ: {distance_actuelle:.2f} km")
    print(f"Nombre de villes: {n_villes}")
    print("Début de la recherche...")

    # 2. Recherche progressive
    etape = 0
    while niveau_exploration > 0.001:
        etape += 1

        # Explorer à ce niveau
        parcours_actuel, distance_actuelle, meilleur_parcours, meilleure_distance = explorer_nouveaux_parcours(
            parcours_actuel, distance_actuelle, meilleur_parcours, meilleure_distance,
            villes, niveau_exploration, essais_par_niveau
        )

        # Afficher progression
        if etape % 10 == 0:
            print(
                f"Niveau {etape:3d} | Exploration: {niveau_exploration:6.1f} | Meilleur: {meilleure_distance:6.2f} km")

        # Réduire l'exploration
        niveau_exploration *= refroidissement

    return meilleur_parcours, meilleure_distance


# Exemple simple avec 5 villes
villes_simples = [
    (0, 0),  # Ville 0
    (10, 10),  # Ville 1
    (10, 0),  # Ville 2
    (0, 10),  # Ville 3
    (5, 5)  # Ville 4
]

print("=== RECHERCHE DU MEILLEUR PARCOURS ===")
parcours_optimal, distance = trouver_meilleur_parcours(villes_simples)

print(f"\n⭐ PARCOURS OPTIMAL TROUVÉ ⭐")
print(f"Ordre des villes: {parcours_optimal}")
print(f"Distance totale: {distance:.2f} km")

# Afficher le parcours détaillé
print("\nDétail du trajet:")
for i in range(len(parcours_optimal)):
    ville_depart = parcours_optimal[i]
    ville_arrivee = parcours_optimal[(i + 1) % len(parcours_optimal)]
    dist = distance_entre_villes(villes_simples[ville_depart], villes_simples[ville_arrivee])
    print(f"  Ville {ville_depart} → Ville {ville_arrivee} : {dist:.2f} km")