# Solution rapide pour le problème conda init

## Problème
`conda init zsh` échoue avec une TypeError sur votre système (bug connu de conda 23.11.0)

## Solution immédiate (sans conda init)

Utilisez cette commande à la place:

```bash
# Au lieu de "conda activate mot", utilisez:
source /Users/jeremy/anaconda3/etc/profile.d/conda.sh && conda activate mot
```

## Tester l'installation

```bash
# 1. Activer l'environnement
source /Users/jeremy/anaconda3/etc/profile.d/conda.sh && conda activate mot

# 2. Vérifier que ça marche
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 3. Installer fsspec (manquant)
pip install fsspec

# 4. Tester MoT
python test_mot.py
```

## Créer un alias pour simplifier

Ajoutez ceci à votre `~/.zshrc`:

```bash
# Ajouter à la fin de ~/.zshrc
alias motenv='source /Users/jeremy/anaconda3/etc/profile.d/conda.sh && conda activate mot'
```

Puis:
```bash
# Recharger le profil
source ~/.zshrc

# Maintenant vous pouvez juste faire:
motenv
```

## Pourquoi ça arrive?

- Conda 23.11.0 a un bug avec `conda init` sur certains systèmes macOS
- La méthode `source conda.sh` contourne ce problème
- C'est une solution permanente et stable

## Commandes utiles

```bash
# Activer mot
source /Users/jeremy/anaconda3/etc/profile.d/conda.sh && conda activate mot

# Désactiver
conda deactivate

# Vérifier l'environnement actif
conda info --envs

# Lister les packages installés
conda list
