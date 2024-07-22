from typing import List, Optional, Tuple
from core.akinator_bayesian import AkinatorBNCore, Answer


class AkinatorTerminalGUI:
    MAX_MULTIPLE_GUESSES = 5
    MAX_ADDITIONAL_QUESTIONS = 5
    CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, csv_file: str, debug: bool = False):
        self.akinator = AkinatorBNCore(csv_file, debug)
        self.user_responses: List[Tuple[str, Optional[Answer]]] = []

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

    def _ask_question(self, feature_name: str) -> Optional[Answer]:
        response_map = {
            Answer.YES: ["sí", "si", "s", "yes", "y", "1"],
            Answer.NO: ["no", "n", "0"],
            Answer.UNCERTAIN: ["no sé", "no se", "ns", "idk", "0.5"],
        }

        while True:
            response = input(f"{feature_name} (si/no/no se): ").strip().lower()
            for answer, options in response_map.items():
                if response in options:
                    return answer
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

        if not top_guesses:
            print(
                "Lo siento, no tengo suficiente información para hacer una suposición."
            )
            self._handle_no_guess()
            return

        character, probability = top_guesses[0]

        if probability >= self.CONFIDENCE_THRESHOLD:
            print(f"¡Creo que he adivinado! ¿Estás pensando en: {character}?")
            print(f"Estoy {probability:.2%} seguro.")
        else:
            print("No estoy muy seguro, pero estas son mis mejores suposiciones:")
            for i, (char, prob) in enumerate(top_guesses, 1):
                print(f"{i}. {char} (Probabilidad: {prob:.2%})")
            print(f"{len(top_guesses) + 1}. Ninguno de los anteriores")

        if self._verify_guess(top_guesses):
            print("¡Excelente! He adivinado correctamente.")
            self._display_character_data(character)
        else:
            print("Lo siento, parece que me equivoqué.")
            self._handle_incorrect_guess()

        print("=" * 60 + "\n")
        self._display_user_responses()

    def _verify_guess(self, top_guesses: List[Tuple[str, float]]) -> bool:
        while True:
            if len(top_guesses) == 1:
                response = (
                    input(
                        f"¿Es correcto que el personaje es {top_guesses[0][0]}? (si/no): "
                    )
                    .strip()
                    .lower()
                )
                if response in ["si", "sí", "s", "yes", "y"]:
                    return True
                elif response in ["no", "n"]:
                    return False
            else:
                try:
                    choice = int(
                        input(
                            "\n¿Cuál de estos personajes estabas pensando? Ingresa el número: "
                        )
                    )
                    if 1 <= choice <= len(top_guesses):
                        return True
                    elif choice == len(top_guesses) + 1:
                        return False
                    print("Por favor, ingresa un número válido.")
                except ValueError:
                    print("Por favor, ingresa un número válido.")

    def _handle_incorrect_guess(self) -> None:
        additional_questions = 0
        while additional_questions < self.MAX_ADDITIONAL_QUESTIONS:
            question = self.akinator.get_next_question()
            if question is None:
                break

            print(
                f"\nPermíteme hacer una pregunta adicional ({additional_questions + 1}/{self.MAX_ADDITIONAL_QUESTIONS}):"
            )
            response = self._ask_question(question)
            self.user_responses.append((question, response))
            self.akinator.process_answer(response)

            top_guesses = self._get_significant_top_guesses(max_guesses=1)
            if top_guesses and self._verify_guess(top_guesses):
                print(f"¡Ahora lo tengo! El personaje es {top_guesses[0][0]}.")
                self._display_character_data(top_guesses[0][0])
                return

            additional_questions += 1

        self._handle_no_guess()

    def _handle_no_guess(self) -> None:
        print(
            "\nNo he podido adivinar el personaje. ¿Podrías decirme en quién estabas pensando?"
        )
        correct_character = input("Nombre del personaje: ").strip()
        print(
            f"Gracias por la información sobre {correct_character}. Lo tendré en cuenta para mejorar en el futuro."
        )
        # Here you would add logic to learn this new character or update the database

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

    def _display_user_responses(self) -> None:
        print("Respuestas del usuario:")
        for feature, response in self.user_responses:
            response_str = response.name if response else "Unknown"
            print(f"{feature}: {response_str}")

    def _display_character_data(self, character: str) -> None:
        print("\nDatos del personaje en el conjunto de datos:")
        print(self.akinator.get_character_data(character))

    def print_tree(self) -> None:
        print("Información del modelo:")
        print(self.akinator.get_tree_info())
