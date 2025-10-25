# Setup Troubleshooting Guide

## Issue: "CondaError: Run 'conda init' before 'conda activate'"

Ce problème arrive quand conda n'est pas correctement initialisé pour votre shell.

### Solution rapide:

```bash
# 1. Initialiser conda pour votre shell (zsh)
conda init zsh

# 2. IMPORTANT: Redémarrer votre terminal ou recharger le profil
source ~/.zshrc

# 3. Activer l'environnement
conda activate mot
```

### Vérifier l'installation:

```bash
# Vérifier que conda fonctionne
conda --version

# Lister les environnements
conda env list

# L'environnement 'mot' devrait apparaître dans la liste
```

---

## Méthode alternative sans conda activate

Si le problème persiste, utilisez cette méthode:

```bash
# Au lieu de:
conda activate mot

# Utilisez:
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mot
```

---

## Installation complète pas-à-pas

### Méthode 1: Environment.yml (recommandé)

```bash
# 1. Créer l'environnement
conda env create -f environment.yml

# 2. Initialiser conda si pas déjà fait
conda init zsh  # ou bash si vous utilisez bash

# 3. Redémarrer le terminal (fermer et rouvrir)
# OU recharger le profil:
source ~/.zshrc  # ou ~/.bashrc

# 4. Activer
conda activate mot

# 5. Vérifier
python -c "import torch; print(f'PyTorch {torch.__version__} ready!')"
python test_mot.py
```

### Méthode 2: Script automatique

```bash
# 1. Rendre le script exécutable
chmod +x setup_env.sh

# 2. Exécuter
./setup_env.sh

# 3. Si erreur "conda activate", redémarrer le terminal puis:
conda activate mot
```

---

## Vérification post-installation

Une fois l'environnement activé, vérifier que tout fonctionne:

```bash
# Activer l'environnement
conda activate mot

# Vérifier Python
python --version
# Devrait afficher: Python 3.10.x

# Vérifier PyTorch
python -c "import torch; print(torch.__version__)"
# Devrait afficher: 2.5.x ou 2.x.x

# Vérifier sympy (doit être 1.13.1 pour compatibilité PyTorch)
python -c "import sympy; print(sympy.__version__)"
# Devrait afficher: 1.13.1

# Vérifier transformers
python -c "import transformers; print(transformers.__version__)"
# Devrait afficher: 4.30.x ou plus

# Test complet du système MoT
python test_mot.py
```

---

## Problèmes courants et solutions

### 1. Conflit de dépendances sympy

**Symptôme**: `ERROR: pip's dependency resolver does not currently take into account...`

**Solution**: Les scripts sont déjà configurés pour installer la bonne version (1.13.1)

### 2. fsspec manquant

**Symptôme**: `torch 2.5.1 requires fsspec, which is not installed`

**Solution**: Les scripts sont déjà configurés pour installer fsspec

### 3. Conda n'est pas trouvé

**Symptôme**: `conda: command not found`

**Solution**: 
```bash
# Installer Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh  # Mac M1/M2/M3
# OU
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  # Linux

# Installer
bash Miniconda3-latest-*.sh

# Suivre les instructions, puis redémarrer le terminal
```

### 4. Permission denied sur setup_env.sh

**Symptôme**: `Permission denied: ./setup_env.sh`

**Solution**:
```bash
chmod +x setup_env.sh
./setup_env.sh
```

---

## Réinstallation propre

Si tout échoue, réinstaller complètement:

```bash
# 1. Supprimer l'environnement existant
conda deactivate
conda env remove -n mot -y

# 2. Nettoyer le cache conda
conda clean --all -y

# 3. Réinstaller avec environment.yml
conda env create -f environment.yml

# 4. Initialiser conda
conda init zsh  # ou bash
source ~/.zshrc  # ou ~/.bashrc

# 5. Activer et tester
conda activate mot
python test_mot.py
```

---

## Aide supplémentaire

Si les problèmes persistent:

1. Vérifier la version de conda: `conda --version` (devrait être ≥ 4.10)
2. Vérifier le shell: `echo $SHELL`
3. Vérifier les variables d'environnement: `echo $CONDA_PREFIX`
4. Consulter les logs conda: `~/.conda/envs/mot/conda-meta/`

Pour plus d'aide, ouvrir une issue sur GitHub avec:
- OS et version
- Version conda (`conda --version`)
- Shell utilisé (`echo $SHELL`)
- Message d'erreur complet
