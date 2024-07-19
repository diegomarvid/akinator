import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import logging
from typing import List, Dict, Tuple

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AkinatorBN:
    def __init__(self, data_path: str):
        """
        Inicializa el Akinator con Redes Bayesianas.

        Parámetros
        ----------
        data_path : str
            Ruta al archivo CSV que contiene el conjunto de datos.
        """
        self.data = pd.read_csv(data_path)
        self.model = None
        self.inference = None
        self.characters = list(self.data["Personaje"])
        self.features = list(self.data.columns[1:])
        logger.info(
            f"AkinatorBN inicializado con {len(self.characters)} personajes y {len(self.features)} características"
        )

    def build_model(self) -> None:
        """
        Construye el modelo de Red Bayesiana basado en el conjunto de datos.
        """
        edges = [("Personaje", feature) for feature in self.features]
        self.model = BayesianNetwork(edges)
        self.model.fit(self.data, estimator=MaximumLikelihoodEstimator)
        logger.info("Modelo de Red Bayesiana construido exitosamente")
        self.inference = VariableElimination(self.model)

    def ask_question(self, feature: str) -> str:
        """
        Genera una pregunta basada en la característica dada.
        """
        return f"¿El personaje {feature.lower()}?"

    def update_probabilities(self, evidence: Dict[str, int]) -> List[Tuple[str, float]]:
        """
        Actualiza las probabilidades de los personajes basándose en la evidencia dada.
        """
        character_probs = []
        for character in self.characters:
            prob = self.calculate_character_probability(character, evidence)
            character_probs.append((character, prob))

        # Normalizar las probabilidades
        total_prob = sum(prob for _, prob in character_probs)
        character_probs = [(char, prob / total_prob) for char, prob in character_probs]

        return sorted(character_probs, key=lambda x: x[1], reverse=True)

    def calculate_character_probability(
        self, character: str, evidence: Dict[str, int]
    ) -> float:
        """
        Calcula la probabilidad de un personaje dado la evidencia.
        """
        character_data = self.data[self.data["Personaje"] == character].iloc[0]
        prob = 1.0
        for feature in self.features:
            if feature in evidence:
                if character_data[feature] == evidence[feature]:
                    prob *= 0.9  # Alta probabilidad si coincide
                else:
                    prob *= 0.1  # Baja probabilidad si no coincide
            else:
                prob *= 0.5  # Probabilidad neutral para características desconocidas
        return prob

    def play(self) -> None:
        """
        Juega al Akinator usando la Red Bayesiana.
        """
        evidence = {}
        for feature in self.features:
            question = self.ask_question(feature)
            while True:
                answer = input(f"{question} (s/n/ns): ").lower()
                if answer in ["s", "n", "ns"]:
                    break
                print("Entrada inválida. Por favor, ingrese 's', 'n', o 'ns'.")

            if answer == "s":
                evidence[feature] = 1
            elif answer == "n":
                evidence[feature] = 0
            # Si la respuesta es 'ns', no la añadimos a la evidencia

            character_probs = self.update_probabilities(evidence)

            logger.info(f"Evidencia actual: {evidence}")
            logger.info("Ranking de probabilidades:")
            for i, (character, prob) in enumerate(character_probs[:10], 1):
                logger.info(f"{i}. {character}: {prob:.2%}")

            top_character, top_prob = character_probs[0]
            second_prob = character_probs[1][1] if len(character_probs) > 1 else 0

            # Verificar si hay un personaje con probabilidad muy alta y significativamente mayor que el segundo
            if top_prob > 0.8 and top_prob > 2 * second_prob:
                print(
                    f"¡Estoy casi seguro! El personaje es {top_character} con {top_prob:.2%} de confianza."
                )
                break

            # Si hemos preguntado todas las características, terminar el juego
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
