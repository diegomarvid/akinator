from typing import List, Tuple, Optional
from core.akinator_bayesian import AkinatorBNCore


class AkinatorTerminalGUI:
    MAX_MULTIPLE_GUESSES = 5

    def __init__(self, csv_file: str, debug: bool = False):
        self.akinator = AkinatorBNCore(csv_file, debug)
        self.user_responses: List[Tuple[str, Optional[bool]]] = []

    def play(self) -> None:
        print("Piensa en un personaje y yo trataré de adivinarlo.")
        self._play_game_loop()
        self._handle_game_end()

    def _play_game_loop(self) -> None:
        while True:
            question = self.akinator.get_next_question()
            if question is None or self.akinator.should_stop():
                break

            response = self._ask_question(question)
            self.user_responses.append((question, response))
            self.akinator.process_answer(response)
            self._display_top_guesses()

    def _ask_question(self, feature_name: str) -> Optional[bool]:
        response_map = {
            True: ["sí", "si", "s", "yes", "y", "1"],
            False: ["no", "n", "0"],
            None: ["no sé", "no se", "ns", "idk", "0.5"],
        }

        while True:
            response = input(f"{feature_name} (si/no/no se): ").strip().lower()
            for value, options in response_map.items():
                if response in options:
                    return value
            print("Por favor, responde con 'sí', 'no', o 'no sé'.")

    def _display_top_guesses(self, n: int = 5) -> None:
        top_guesses = self.akinator.get_top_guesses(n=n)
        formatted_guesses = "\n".join(
            f"{i+1}. {name}: {score:.2f}" for i, (name, score) in enumerate(top_guesses)
        )
        self.akinator.logger.info(f"\nTop guesses:\n{formatted_guesses}\n")

    def _handle_game_end(self) -> None:
        top_guesses = self._get_significant_top_guesses(
            max_guesses=self.MAX_MULTIPLE_GUESSES
        )
        print("\n" + "=" * 60)

        character, probability = (
            self._handle_multiple_top_guesses(top_guesses)
            if self._has_multiple_top_guesses(top_guesses)
            else top_guesses[0]
        )

        print(f"¡He adivinado! Creo que estás pensando en: {character}")
        print(f"Estoy {probability:.2%} seguro.")
        print("=" * 60 + "\n")

        self._display_user_responses()
        self._display_character_data(character)

    def _has_multiple_top_guesses(self, top_guesses: List[Tuple[str, float]]) -> bool:
        return len(top_guesses) > 1 and top_guesses[0][1] == top_guesses[1][1]

    def _get_significant_top_guesses(
        self, relative_threshold: float = 0.9, max_guesses: int = 10
    ) -> List[Tuple[str, float]]:
        all_guesses = self.akinator.get_top_guesses()
        if not all_guesses:
            return []

        top_probability = all_guesses[0][1]
        threshold = top_probability * relative_threshold

        significant_guesses = [guess for guess in all_guesses if guess[1] >= threshold]

        return significant_guesses[:max_guesses]

    def _handle_multiple_top_guesses(
        self, top_guesses: List[Tuple[str, float]]
    ) -> Tuple[str, float]:
        print(
            "No puedo estar completamente seguro, pero creo que estás pensando en uno de estos personajes:"
        )
        for i, (character, probability) in enumerate(top_guesses, 1):
            print(f"{i}. {character} (Probabilidad: {probability:.2%})")

        while True:
            try:
                choice = int(
                    input(
                        "\n¿Cuál de estos personajes estabas pensando? Ingresa el número: "
                    )
                )
                if 1 <= choice <= len(top_guesses):
                    return top_guesses[choice - 1]
                print("Por favor, ingresa un número válido.")
            except ValueError:
                print("Por favor, ingresa un número válido.")

    def _display_user_responses(self) -> None:
        print("Respuestas del usuario:")
        for feature, response in self.user_responses:
            response_str = (
                "Sí" if response else "No" if response is not None else "No sé"
            )
            print(f"{feature}: {response_str}")

    def _display_character_data(self, character: str) -> None:
        print("\nDatos del personaje en el conjunto de datos:")
        print(self.akinator.get_character_data(character))

    def print_tree(self) -> None:
        print("Información del modelo:")
        print(self.akinator.get_tree_info())
