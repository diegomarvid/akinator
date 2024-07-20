import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import logging
from typing import List, Dict, Tuple
from functools import lru_cache

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

NEGATIVE_EVIDENCE = 0
POSITIVE_EVIDENCE = 1
UNCERTAIN_EVIDENCE = 0.5  # New constant for uncertain responses


class AkinatorBN:
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)
        self.model = None
        self.inference = None
        self.characters = list(self.data["Personaje"])
        self.features = list(self.data.columns[1:])
        logger.info(
            f"AkinatorBN inicializado con {len(self.characters)} personajes y {len(self.features)} características"
        )

    def build_model(self) -> None:
        edges = [("Personaje", feature) for feature in self.features]
        self.model = BayesianNetwork(edges)
        self.model.fit(self.data, estimator=MaximumLikelihoodEstimator)
        logger.info("Modelo de Red Bayesiana construido exitosamente")
        self.inference = VariableElimination(self.model)

    def ask_question(self, feature: str) -> str:
        return f"¿El personaje {feature.lower()}?"

    @lru_cache(maxsize=None)
    def _cached_update_probabilities(
        self, evidence: Tuple[Tuple[str, float], ...]
    ) -> List[Tuple[str, float]]:
        return self.update_probabilities(dict(evidence))

    def update_probabilities(
        self, evidence: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        logger.debug(f"Updating probabilities with evidence: {evidence}")

        # Handle uncertain evidence
        certain_evidence = {
            k: v for k, v in evidence.items() if v != UNCERTAIN_EVIDENCE
        }
        uncertain_features = [k for k, v in evidence.items() if v == UNCERTAIN_EVIDENCE]

        query = self.inference.query(variables=["Personaje"], evidence=certain_evidence)
        probs_dict = query.values

        # Create a mapping of character names to their corresponding probabilities
        character_probs = {
            str(char): prob
            for char, prob in zip(query.state_names["Personaje"], probs_dict)
        }

        # Ensure all characters are in the dictionary, even if with zero probability
        for char in self.characters:
            character_probs.setdefault(char, 0.0)

        # Adjust probabilities for uncertain features
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

        # Calculate total probability and normalize in one pass
        total_prob = sum(character_probs.values())
        normalized_probs = [
            (char, prob / total_prob) for char, prob in character_probs.items()
        ]

        # Sort probabilities in descending order
        sorted_probs = sorted(normalized_probs, key=lambda x: x[1], reverse=True)

        logger.debug(f"Updated probabilities: {sorted_probs}")

        return sorted_probs

    def _calculate_entropy(self, probs: List[Tuple[str, float]]) -> float:
        entropy = -sum(p * np.log2(p) for _, p in probs if p > 0)
        logger.debug(f"Calculated entropy: {entropy}")
        return entropy

    def calculate_information_gain(
        self, feature: str, evidence: Dict[str, float]
    ) -> float:
        logger.debug(f"Calculating information gain for feature: {feature}")
        logger.debug(f"Current evidence: {evidence}")

        current_probs = self._cached_update_probabilities(
            tuple(sorted(evidence.items()))
        )
        logger.debug(f"Current probabilities: {current_probs}")

        current_entropy = self._calculate_entropy(current_probs)
        logger.debug(f"Current entropy: {current_entropy}")

        entropy_yes = entropy_no = entropy_uncertain = 0
        prob_yes = prob_no = prob_uncertain = 0

        for answer in [NEGATIVE_EVIDENCE, POSITIVE_EVIDENCE, UNCERTAIN_EVIDENCE]:
            new_evidence = {**evidence, feature: answer}
            logger.debug(f"New evidence for {feature}={answer}: {new_evidence}")

            probs = self._cached_update_probabilities(
                tuple(sorted(new_evidence.items()))
            )
            logger.debug(f"Probabilities for {feature}={answer}: {probs}")

            entropy = self._calculate_entropy(probs)
            logger.debug(f"Entropy for {feature}={answer}: {entropy}")

            prob_sum = sum(p for _, p in probs if p > 0)
            logger.debug(f"Probability sum for {feature}={answer}: {prob_sum}")

            if answer == NEGATIVE_EVIDENCE:
                entropy_no, prob_no = entropy, prob_sum
            elif answer == POSITIVE_EVIDENCE:
                entropy_yes, prob_yes = entropy, prob_sum
            else:
                entropy_uncertain, prob_uncertain = entropy, prob_sum

        total_prob = prob_yes + prob_no + prob_uncertain
        logger.debug(f"Total probability: {total_prob}")

        if total_prob > 0:
            prob_yes /= total_prob
            prob_no /= total_prob
            prob_uncertain /= total_prob
        else:
            prob_yes = prob_no = prob_uncertain = 1 / 3

        logger.debug(
            f"Normalized probabilities: yes={prob_yes}, no={prob_no}, uncertain={prob_uncertain}"
        )

        info_gain = current_entropy - (
            prob_yes * entropy_yes
            + prob_no * entropy_no
            + prob_uncertain * entropy_uncertain
        )
        logger.debug(f"Information gain for {feature}: {info_gain}")

        return info_gain

    def select_best_feature(self, evidence: Dict[str, float]) -> str:
        remaining_features = [f for f in self.features if f not in evidence]
        info_gains = [
            (f, self.calculate_information_gain(f, evidence))
            for f in remaining_features
        ]

        ranked_features = sorted(info_gains, key=lambda x: x[1], reverse=True)

        print("\nRanking of features by Information Gain:")
        for rank, (feature, info_gain) in enumerate(ranked_features, 1):
            print(f"Rank {rank}: Feature: {feature}, Information Gain: {info_gain:.2f}")
        print()

        return ranked_features[0][0]

    def play(self) -> None:
        evidence = {}
        while True:
            feature = self.select_best_feature(evidence)
            question = self.ask_question(feature)
            while True:
                answer = input(f"{question} (s/n/ns): ").lower()
                if answer in ["s", "n", "ns"]:
                    break
                print("Entrada inválida. Por favor, ingrese 's', 'n', o 'ns'.")

            if answer == "s":
                evidence[feature] = POSITIVE_EVIDENCE
            elif answer == "n":
                evidence[feature] = NEGATIVE_EVIDENCE
            else:  # "ns"
                evidence[feature] = UNCERTAIN_EVIDENCE

            character_probs = self._cached_update_probabilities(
                tuple(sorted(evidence.items()))
            )

            logger.info(f"Evidencia actual: {evidence}")
            logger.info("Ranking de probabilidades:")
            for i, (character, prob) in enumerate(character_probs[:10], 1):
                logger.info(f"{i}. {character}: {prob:.2%}")

            top_character, top_prob = character_probs[0]
            second_prob = character_probs[1][1] if len(character_probs) > 1 else 0

            if top_prob > 0.8 and top_prob > 2 * second_prob:
                print(
                    f"¡Estoy casi seguro! El personaje es {top_character} con {top_prob:.2%} de confianza."
                )
                break

            if len(evidence) == len(self.features):
                print(
                    f"Creo que el personaje es {top_character} con {top_prob:.2%} de confianza."
                )
                break

        print("Juego terminado. Aquí están las probabilidades finales:")
        for character, prob in character_probs:
            print(f"{character}: {prob:.2%}")


if __name__ == "__main__":
    akinator = AkinatorBN("data/personajes.csv")
    akinator.build_model()
    akinator.play()
