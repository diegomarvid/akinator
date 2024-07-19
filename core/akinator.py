import logging
from typing import List, Optional
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


class AkinatorCore:
    """
    Core logic for a simplified version of the Akinator game using a decision tree classifier.

    This class implements the game logic where the computer tries to guess
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
    current_node : int
        The current node in the decision tree.
    """

    def __init__(self, csv_file: str, debug: bool = False) -> None:
        self.df: pd.DataFrame = pd.read_csv(csv_file)
        self.le: LabelEncoder = LabelEncoder()
        self.clf: DecisionTreeClassifier = DecisionTreeClassifier(random_state=42)
        self.feature_names: List[str] = []
        self.current_node: int = 0
        self._prepare_model()
        self.logger: logging.Logger = self._setup_logger(debug)

    def _setup_logger(self, debug: bool) -> logging.Logger:
        """
        Set up and configure the logger.

        Parameters
        ----------
        debug : bool
            If True, sets logging level to DEBUG.

        Returns
        -------
        logging.Logger
            Configured logger instance.
        """
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
        Prepare and train the decision tree model.

        This method separates features and target from the dataframe,
        encodes the target, fits the classifier, and stores feature names.
        """
        X = self.df.drop(columns=["Personaje"])
        y = self.df["Personaje"]
        y_encoded = self.le.fit_transform(y)
        self.clf.fit(X, y_encoded)
        self.feature_names = X.columns.tolist()

    def get_next_question(self) -> Optional[str]:
        """
        Get the next question to ask based on the current node in the decision tree.

        Returns
        -------
        Optional[str]
            The feature name to ask about, or None if we've reached a leaf node.
        """
        feature_index: int = self.clf.tree_.feature[self.current_node]
        if feature_index != -2:  # Not a leaf
            return self.feature_names[feature_index]
        return None

    def process_answer(self, answer: bool) -> None:
        """
        Process the user's answer and move to the next node in the decision tree.

        Parameters
        ----------
        answer : bool
            True if the user's response is affirmative, False otherwise.
        """
        if answer:
            self.current_node = self.clf.tree_.children_right[self.current_node]
        else:
            self.current_node = self.clf.tree_.children_left[self.current_node]

    def make_guess(self) -> str:
        """
        Make a guess about the character based on the current position in the decision tree.

        Returns
        -------
        str
            The guessed character name.
        """
        pred: int = self.clf.tree_.value[self.current_node].argmax()
        return self.le.inverse_transform([pred])[0]

    def reset_game(self) -> None:
        """
        Reset the game state to start a new game.
        """
        self.current_node = 0

    def get_tree_info(self) -> str:
        """
        Get a string representation of the decision tree.

        Returns
        -------
        str
            String representation of the decision tree.
        """
        from sklearn.tree import export_text

        return export_text(self.clf, feature_names=self.feature_names)

    def log_debug_info(self, feature_name: str, response: bool) -> None:
        """
        Log debug information about the current game state.

        Parameters
        ----------
        feature_name : str
            The name of the feature being asked about.
        response : bool
            The user's response to the question.
        """
        self.logger.debug(
            f"Node {self.current_node}, Feature: {feature_name}, Response: {response}"
        )

    def get_character_data(self, character: str) -> pd.Series:
        """
        Get the data for a specific character from the dataset.

        Parameters
        ----------
        character : str
            The name of the character to retrieve data for.

        Returns
        -------
        pd.Series
            The row of data for the specified character.
        """
        return self.df[self.df["Personaje"] == character].iloc[0]
