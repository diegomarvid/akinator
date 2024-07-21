import logging
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork


class Answer(Enum):
    NO = 0.0
    YES = 1.0
    UNCERTAIN = 0.5

    @classmethod
    def from_value(cls, value: float) -> "Answer":
        for answer in cls:
            if (
                abs(answer.value - value) < 1e-6
            ):  # Use a small epsilon for float comparison
                return answer
        raise ValueError(f"No matching Answer for value: {value}")


class AkinatorBNCore:
    def __init__(self, csv_file: str, debug: bool = False):
        self.data = pd.read_csv(csv_file)
        self.model = None
        self.inference = None
        self.characters = list(self.data["Personaje"])
        self.features = self._get_valid_features()
        self.evidence: Dict[str, Answer] = {}
        self.current_feature = None
        self.logger = self._setup_logger(debug)
        self._prepare_model()
        self.confidence_threshold = 0.8
        self.max_questions = 20

    def _setup_logger(self, debug: bool) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG if debug else logging.INFO)

        # Clear existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Prevent the logger from propagating messages to the root logger
        logger.propagate = False

        return logger

    def _get_valid_features(self) -> List[str]:
        return [col for col in self.data.columns[1:] if self.data[col].nunique() > 1]

    def _prepare_model(self) -> None:
        valid_columns = ["Personaje"] + self.features
        filtered_data = self.data[valid_columns]

        edges = [("Personaje", feature) for feature in self.features]
        self.model = BayesianNetwork(edges)
        self.model.fit(filtered_data, estimator=MaximumLikelihoodEstimator)
        self.inference = VariableElimination(self.model)
        self.logger.info("Bayesian Network model built successfully")

    def get_next_question(self) -> Optional[str]:
        if len(self.evidence) >= self.max_questions:
            return None
        self.current_feature = self.select_best_feature(self.evidence)
        if self.current_feature:
            return f"¿El personaje {self.current_feature.lower()}?"
        return None

    def process_answer(self, answer: Optional[Answer]) -> None:
        if self.current_feature:
            if answer is None:
                self.evidence[self.current_feature] = Answer.UNCERTAIN
            else:
                self.evidence[self.current_feature] = answer
            self.log_debug_info(self.current_feature, answer)

    def _evidence_to_float(self, evidence: Dict[str, Answer]) -> Dict[str, float]:
        return {k: v.value for k, v in evidence.items()}

    def make_guess(self) -> Tuple[str, float]:
        probs = self.update_probabilities(tuple(sorted(self.evidence.items())))
        top_character, top_prob = probs[0]
        return top_character, top_prob

    def get_top_guesses(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top N character guesses with their probabilities.

        Parameters
        ----------
        n : int, optional
            Number of top guesses to return (default is 10)

        Returns
        -------
        List[Tuple[str, float]]
            List of tuples containing character names and their probabilities,
            sorted in descending order of probability.
        """
        probs = self.update_probabilities(tuple(sorted(self.evidence.items())))
        return probs[:n]

    def should_stop(self) -> bool:
        probs = self.update_probabilities(tuple(sorted(self.evidence.items())))
        top_prob = probs[0][1]
        second_prob = probs[1][1] if len(probs) > 1 else 0
        return top_prob > self.confidence_threshold and top_prob > 2 * second_prob

    def reset_game(self) -> None:
        self.evidence = {}
        self.current_feature = None

    @lru_cache(maxsize=None)
    def update_probabilities(
        self, evidence: Tuple[Tuple[str, Answer], ...]
    ) -> Tuple[Tuple[str, float], ...]:
        evidence_dict = dict(evidence)
        float_evidence = self._evidence_to_float(evidence_dict)
        certain_evidence = {
            k: v for k, v in float_evidence.items() if v != Answer.UNCERTAIN.value
        }
        uncertain_features = [
            k for k, v in float_evidence.items() if v == Answer.UNCERTAIN.value
        ]

        query = self.inference.query(variables=["Personaje"], evidence=certain_evidence)
        probs_dict = query.values
        character_probs = {
            str(char): prob
            for char, prob in zip(query.state_names["Personaje"], probs_dict)
        }

        for char in self.characters:
            character_probs.setdefault(char, 0.0)

        for feature in uncertain_features:
            feature_probs = self.inference.query(
                variables=[feature], evidence=certain_evidence
            )
            for char in self.characters:
                char_prob = character_probs[char]
                feature_prob = feature_probs.values[
                    1
                ]  # Probability of feature being True
                character_probs[char] = char_prob * (0.5 + 0.5 * feature_prob)

        total_prob = sum(character_probs.values())
        normalized_probs = [
            (char, prob / total_prob) for char, prob in character_probs.items()
        ]
        return tuple(sorted(normalized_probs, key=lambda x: x[1], reverse=True))

    def select_best_feature(self, evidence: Dict[str, float]) -> Optional[str]:
        remaining_features = [f for f in self.features if f not in evidence]
        if not remaining_features:
            return None
        evidence_tuple = tuple(sorted(evidence.items()))
        info_gains = [
            (f, self.calculate_information_gain(f, evidence_tuple))
            for f in remaining_features
        ]
        return max(info_gains, key=lambda x: x[1])[0]

    @lru_cache(maxsize=None)
    def calculate_information_gain(
        self, feature: str, evidence: Tuple[Tuple[str, Answer], ...]
    ) -> float:
        evidence_dict = dict(evidence)
        current_probs = self.update_probabilities(evidence)
        current_entropy = self._calculate_entropy(current_probs)

        entropy_yes = entropy_no = entropy_uncertain = 0
        prob_yes = prob_no = prob_uncertain = 0

        for answer in Answer:
            new_evidence = tuple(sorted({**evidence_dict, feature: answer}.items()))
            probs = self.update_probabilities(new_evidence)
            entropy = self._calculate_entropy(probs)
            prob_sum = sum(p for _, p in probs)

            if answer == Answer.NO:
                entropy_no, prob_no = entropy, prob_sum
            elif answer == Answer.YES:
                entropy_yes, prob_yes = entropy, prob_sum
            else:
                entropy_uncertain, prob_uncertain = entropy, prob_sum

        total_prob = prob_yes + prob_no + prob_uncertain
        if total_prob > 0:
            prob_yes /= total_prob
            prob_no /= total_prob
            prob_uncertain /= total_prob
        else:
            prob_yes = prob_no = prob_uncertain = 1 / 3

        info_gain = current_entropy - (
            prob_yes * entropy_yes
            + prob_no * entropy_no
            + prob_uncertain * entropy_uncertain
        )

        return info_gain

    @staticmethod
    @lru_cache(maxsize=1000)
    def _calculate_entropy(probs: Tuple[Tuple[str, float], ...]) -> float:
        return -sum(p * np.log2(p) for _, p in probs if p > 0)

    def log_debug_info(self, feature_name: str, response: Optional[Answer]) -> None:
        response_str = "Unknown" if response is None else response.name
        self.logger.debug(f"Question: {feature_name}, Answer: {response_str}")
        self.logger.debug(f"Current evidence: {self.evidence}")

    def get_character_data(self, character: str) -> pd.Series:
        character_data = self.data[self.data["Personaje"] == character]
        if character_data.empty:
            raise KeyError(f"Character '{character}' not found in the dataset")
        return character_data.iloc[0]

    def get_tree_info(self) -> str:
        return "Bayesian Network does not have a tree representation."
