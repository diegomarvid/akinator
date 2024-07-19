import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import streamlit as st
from typing import List, Tuple, Dict, Any
from core import AkinatorCore

CSV_FILE = "data/personajes.csv"  # Replace with the path to your CSV file


@st.cache(allow_output_mutation=True)
def get_akinator_core() -> AkinatorCore:
    """
    Create and cache an instance of AkinatorCore.

    Returns
    -------
    AkinatorCore
        An instance of the AkinatorCore class.

    Notes
    -----
    This function is cached by Streamlit to avoid recreating the AkinatorCore
    instance on every rerun.
    """
    return AkinatorCore(CSV_FILE, debug=True)


def display_user_data(user_responses: List[Tuple[str, bool]]) -> None:
    """
    Display the user's responses in the sidebar.

    Parameters
    ----------
    user_responses : List[Tuple[str, bool]]
        A list of tuples containing the questions and the user's responses.

    Returns
    -------
    None
    """
    st.sidebar.write("Respuestas del usuario:")
    for feature, response in user_responses:
        st.sidebar.write(f"{feature}: {'Sí' if response else 'No'}")


def main() -> None:
    """
    Main function to run the Streamlit Akinator game.

    This function sets up the Streamlit interface, handles user interactions,
    and manages the game state.

    Returns
    -------
    None
    """
    st.title("Simplified Akinator Game")

    akinator_core = get_akinator_core()

    if "game_state" not in st.session_state:
        st.session_state.game_state: Dict[str, Any] = {
            "user_responses": [],
            "game_over": False,
            "current_node": 0,
            "show_user_data": False,
            "show_decision_tree": False,
        }

    # Sidebar controls
    st.sidebar.title("Opciones")
    st.session_state.game_state["show_decision_tree"] = st.sidebar.checkbox(
        "Mostrar árbol de decisión",
        value=st.session_state.game_state.get("show_decision_tree", False),
    )
    st.session_state.game_state["show_user_data"] = st.sidebar.checkbox(
        "Mostrar datos del usuario",
        value=st.session_state.game_state.get("show_user_data", False),
    )

    if st.session_state.game_state["show_decision_tree"]:
        st.sidebar.text(akinator_core.get_tree_info())

    if st.session_state.game_state["show_user_data"]:
        display_user_data(st.session_state.game_state["user_responses"])

    if not st.session_state.game_state["game_over"]:
        st.write("Piensa en un personaje y yo trataré de adivinarlo.")
        question = akinator_core.get_next_question()
        if question:
            response = st.radio(
                f"¿{question}?",
                ("Sí", "No"),
                key=f"question_{st.session_state.game_state['current_node']}",
            )
            if st.button(
                "Siguiente", key=f"next_{st.session_state.game_state['current_node']}"
            ):
                answer = response == "Sí"
                st.session_state.game_state["user_responses"].append((question, answer))
                akinator_core.process_answer(answer)
                akinator_core.log_debug_info(question, answer)
                st.session_state.game_state["current_node"] = akinator_core.current_node
                st.experimental_rerun()
        else:
            character = akinator_core.make_guess()
            st.success(f"¡He adivinado! Creo que estás pensando en: {character}")
            st.session_state.game_state["game_over"] = True

    if st.button("Reiniciar juego"):
        st.session_state.game_state = {
            "user_responses": [],
            "game_over": False,
            "current_node": 0,
            "show_user_data": st.session_state.game_state["show_user_data"],
            "show_decision_tree": st.session_state.game_state["show_decision_tree"],
        }
        akinator_core.reset_game()
        st.experimental_rerun()


if __name__ == "__main__":
    main()
