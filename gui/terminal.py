from typing import List, Tuple, Optional
from core.akinator_bayesian import AkinatorBNCore


class AkinatorTerminalGUI:
    def __init__(self, csv_file: str, debug: bool = False):
        self.akinator = AkinatorBNCore(csv_file, debug)
        self.user_responses: List[Tuple[str, Optional[bool]]] = []

    def _ask_question(self, feature_name: str) -> Optional[bool]:
        while True:
            response = input(f"¿{feature_name}? (sí/no/no sé): ").strip().lower()
            if response in ["sí", "si", "s", "yes", "y"]:
                return True
            elif response in ["no", "n"]:
                return False
            elif response in ["no sé", "no se", "ns", "idk"]:
                return None
            else:
                print("Por favor, responde con 'sí', 'no', o 'no sé'.")

    def play(self) -> None:
        print("Piensa en un personaje y yo trataré de adivinarlo.")

        while True:
            question = self.akinator.get_next_question()
            if question is None:
                break

            response = self._ask_question(question)
            self.user_responses.append((question, response))
            self.akinator.process_answer(response)
            self.akinator.log_debug_info(question, response)

            if self.akinator.should_stop():
                break

        character, probability = self.akinator.make_guess()

        print("\n" + "=" * 60)
        print(f"¡He adivinado! Creo que estás pensando en: {character}")
        print(f"Estoy {probability:.2%} seguro.")
        print("=" * 60 + "\n")

        print("Respuestas del usuario:")
        for feature, response in self.user_responses:
            response_str = (
                "Sí" if response else "No" if response is not None else "No sé"
            )
            print(f"{feature}: {response_str}")

        print("\nDatos del personaje en el conjunto de datos:")
        character_data = self.akinator.get_character_data(character)
        print(character_data)

    def print_tree(self) -> None:
        print("Información del modelo:")
        print(self.akinator.get_tree_info())
