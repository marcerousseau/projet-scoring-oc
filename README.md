# project-scoring
Ce fichier présente la structure du code du projet

# Code structure is as follow

- app
  - static -> storing all static assets (including css)
  - templates -> storing templates to be rendered
    - landing.html -> template de la home page
  - tests -> storing all the tests (unit tests, integration tests, functional tests)
  - best_model_2.pkl : le pickle du modèle utilisé en production
  - run.py -> entry point to run the flask server
  - site_routes.py -> les différentes routes du server (API, html page)
  - wsgi.py -> sert à créer le serveur
- feature_engineering.py -> permet de réaliser la partie feature engineering à partir des données fournies. Nous concaténons les différentes sources et les augmentons afin d'obtenir 795 features
- LICENCE -> Licence du projet (MIT)
- DockerFile -> Pour containeriser l'application
- model_exploration.py: pour entrainer le modèle en tunant les hyperparamêtres par corss-validation et grid search. Utilise ML Flow.
- model_fast_exploration.py: pour réaliser une première étude des modèles avec lazy predict en incluant un dummy classifier.
- README.md -> ce fichier
- requirements.txt -> L'ensemble de requirement pour pip

# Installation
First we need to clone the repo locally
```sh
git clone git@github.com:marcerousseau/projet-scoring-oc.git
```

Then create a venv and install the requirements
```sh
python3 -m venv ./venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

Then go into the app folder and run the server
```sh
cd app
python3 run.py
```

# DEMO
Demo is accessible here : https://project-scoring-jltkqgajta-ew.a.run.app