# akinator_terminal_gui.py

from typing import List, Tuple
from akinator_core import AkinatorCore

class AkinatorTerminalGUI:
    """
    Terminal-based GUI for the Akinator game.

    This class manages the terminal interface for the Akinator game,
    handling user interactions and display logic.

    Parameters
    ----------
    akinator : AkinatorCore
        An instance of the AkinatorCore class to handle game logic.

    Attributes
    ----------
    akinator : AkinatorCore
        The AkinatorCore instance used for game logic.
    user_responses : List[Tuple[str, bool]]
        List of user responses (feature name and boolean response).
    """

    def __init__(self, akinator: AkinatorCore):
        self.akinator = akinator
        self.user_responses: List[Tuple[str, bool]] = []

    def _ask_question(self, feature_name: str) -> bool:
        """
        Ask a yes/no question to the user about a given feature.

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
        Start and run the Akinator game in the terminal.

        This method walks through the decision tree, asking questions
        based on features until it reaches a leaf node and makes a guess.
        It then displays the result and logs debug information.
        """
        print("Piensa en un personaje y yo trataré de adivinarlo.")
        
        while True:
            question = self.akinator.get_next_question()
            if question is None:
                break

            response = self._ask_question(question)
            self.user_responses.append((question, response))
            self.akinator.process_answer(response)
            self.akinator.log_debug_info(question, response)

        character = self.akinator.make_guess()

        print("\n" + "=" * 60)
        print(f"¡He adivinado! Creo que estás pensando en: {character}")
        print("=" * 60 + "\n")

        print("Respuestas del usuario:")
        for feature, response in self.user_responses:
            print(f"{feature}: {'Sí' if response else 'No'}")

        print("\nDatos del personaje en el conjunto de datos:")
        character_data = self.akinator.get_character_data(character)
        print(character_data)

    def print_tree(self) -> None:
        """
        Print the decision tree rules.
        """
        print("Árbol de decisión:")
        print(self.akinator.get_tree_info())

def create_akinator_terminal_gui(csv_file: str, debug: bool = False) -> AkinatorTerminalGUI:
    """
    Create and return an instance of AkinatorTerminalGUI.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing character data.
    debug : bool, optional
        If True, enables debug logging. Default is False.

    Returns
    -------
    AkinatorTerminalGUI
        An instance of the AkinatorTerminalGUI class.
    """
    akinator_core = AkinatorCore(csv_file, debug)
    return AkinatorTerminalGUI(akinator_core)