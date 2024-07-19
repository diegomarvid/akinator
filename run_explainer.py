# %%
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import pandas as pd
from akinator_core import AkinatorCore

# Load your data
csv_file = "data/personajes.csv"
akinator = AkinatorCore(csv_file)

# Encode the target variable
y_encoded = akinator.le.transform(akinator.df["Personaje"])

# Prepare the Explainer
explainer = ClassifierExplainer(
    model=akinator.clf,
    X=akinator.df.drop(columns=["Personaje"]),
    y=y_encoded,
    labels=akinator.le.classes_.tolist(),
    shap="tree",
)

# Launch the dashboard
ExplainerDashboard(explainer).run()

# %%
