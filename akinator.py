import logging
from typing import List

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


class SimplifiedAkinator:
    """
    A simplified version of the Akinator game using a decision tree classifier.

    This class implements a guessing game where the computer tries to guess
    a character based on yes/no questions.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing character data.
    debug : bool, optional
        If True, sets logging level to DEBUG. Default is False.

    Attributes
    ----------
    df : pd.DataFrame
        The dataframe containing the character data.
    le : LabelEncoder
        Label encoder for character names.
    clf : DecisionTreeClassifier
        The decision tree classifier used for guessing.
    feature_names : List[str]
        List of feature names used in the decision tree.
    logger : logging.Logger
        Logger for debug and info messages.
    """

    def __init__(self, csv_file: str, debug: bool = False) -> None:
        self.df = pd.read_csv(csv_file)
        self.le = LabelEncoder()
        self.clf = DecisionTreeClassifier(random_state=42)
        self.feature_names: List[str] = []
        self._prepare_model()
        self.logger = self._setup_logger(debug)

    def _setup_logger(self, debug: bool) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _prepare_model(self) -> None:
        """
        Prepares and trains the decision tree model.

        This method separates features and target from the dataframe,
        encodes the target, fits the classifier, and stores feature names.
        """
        X = self.df.drop(columns=["Personaje"])
        y = self.df["Personaje"]
        y_encoded = self.le.fit_transform(y)
        self.clf.fit(X, y_encoded)
        self.feature_names = X.columns.tolist()

    def _ask_question(self, feature_name: str) -> bool:
        """
        Asks a yes/no question to the user about a given feature.

        Parameters
        ----------
        feature_name : str
            The name of the feature to ask about.

        Returns
        -------
        bool
            True if the user's response is affirmative, False otherwise.
        """
        while True:
            response = input(f"¿{feature_name}? (sí/no): ").strip().lower()
            if response in ["sí", "si", "s", "yes", "y"]:
                return True
            elif response in ["no", "n"]:
                return False
            else:
                print("Por favor, responde con 'sí' o 'no'.")

    def play(self) -> None:
        """
        Starts and runs the Akinator game.

        This method walks through the decision tree, asking questions
        based on features until it reaches a leaf node and makes a guess.
        It then displays the result and logs debug information.
        """
        print("Piensa en un personaje y yo trataré de adivinarlo.")
        curr_node = 0
        user_responses = []

        while self.clf.tree_.feature[curr_node] != -2:  # Not a leaf
            feature_index = self.clf.tree_.feature[curr_node]
            feature_name = self.feature_names[feature_index]

            response = self._ask_question(feature_name)
            user_responses.append((feature_name, response))

            if response:
                curr_node = self.clf.tree_.children_right[curr_node]
            else:
                curr_node = self.clf.tree_.children_left[curr_node]

            self.logger.debug(
                f"Node {curr_node}, Feature: {feature_name}, Response: {response}"
            )

        pred = self.clf.tree_.value[curr_node].argmax()
        character = self.le.inverse_transform([pred])[0]

        print("\n" + "=" * 60)
        print(f"¡He adivinado! Creo que estás pensando en: {character}")
        print("=" * 60 + "\n")

        self.logger.debug("User responses:")
        for feature, response in user_responses:
            self.logger.debug(f"{feature}: {'Sí' if response else 'No'}")

        self.logger.debug("Character data in dataset:")
        self.logger.debug(self.df[self.df["Personaje"] == character])

    def print_tree(self) -> None:
        """
        Prints the decision tree rules.

        This method uses sklearn's export_text function to generate
        a text representation of the decision tree and logs it.
        """
        from sklearn.tree import export_text

        tree_rules = export_text(self.clf, feature_names=self.feature_names)
        self.logger.info("Decision Tree Rules:")
        self.logger.info(tree_rules)
