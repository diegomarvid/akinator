import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from core import AkinatorBNCore, Answer


@st.cache(allow_output_mutation=True)
def get_akinator_core() -> AkinatorBNCore:
    """
    Create and cache an instance of AkinatorBNCore.
    """
    return AkinatorBNCore(csv_file="data/personajes_expanded.csv", debug=True)


def display_user_data(user_responses: List[Tuple[str, Optional[Answer]]]) -> None:
    """
    Display the user's responses in the sidebar.
    """
    st.sidebar.write("Respuestas del usuario:")
    for feature, response in user_responses:
        response_str = response.name if response else "No sé"
        st.sidebar.write(f"{feature}: {response_str}")


def main() -> None:
    """
    Main function to run the Streamlit Akinator game.
    """
    st.title("Akinator Game (Bayesian Network Version)")

    akinator_core = get_akinator_core()

    if "game_state" not in st.session_state:
        st.session_state.game_state: Dict[str, Any] = {
            "user_responses": [],
            "game_over": False,
            "show_user_data": False,
            "show_model_info": False,
        }

    # Sidebar controls
    st.sidebar.title("Opciones")
    st.session_state.game_state["show_model_info"] = st.sidebar.checkbox(
        "Mostrar información del modelo",
        value=st.session_state.game_state.get("show_model_info", False),
    )
    st.session_state.game_state["show_user_data"] = st.sidebar.checkbox(
        "Mostrar datos del usuario",
        value=st.session_state.game_state.get("show_user_data", False),
    )

    if st.session_state.game_state["show_model_info"]:
        st.sidebar.text(akinator_core.get_tree_info())

    if st.session_state.game_state["show_user_data"]:
        display_user_data(st.session_state.game_state["user_responses"])

    if not st.session_state.game_state["game_over"]:
        st.write("Piensa en un personaje y yo trataré de adivinarlo.")
        question = akinator_core.get_next_question()
        if question:
            response = st.radio(
                question,
                ("Sí", "No", "No sé"),
                key=f"question_{len(st.session_state.game_state['user_responses'])}",
            )
            if st.button(
                "Siguiente",
                key=f"next_{len(st.session_state.game_state['user_responses'])}",
            ):
                answer = (
                    Answer.YES
                    if response == "Sí"
                    else Answer.NO if response == "No" else Answer.UNCERTAIN
                )
                st.session_state.game_state["user_responses"].append((question, answer))
                akinator_core.process_answer(answer)

                if akinator_core.should_stop():
                    st.session_state.game_state["game_over"] = True

                st.experimental_rerun()
        else:
            st.session_state.game_state["game_over"] = True
            st.experimental_rerun()

    if st.session_state.game_state["game_over"]:
        top_guesses = akinator_core.get_top_guesses(10)  # Get top 10 guesses

        st.success(f"¡He adivinado! Aquí están mis mejores suposiciones:")

        # Create a dataframe for the top guesses
        df = pd.DataFrame(top_guesses, columns=["Personaje", "Probabilidad"])
        df["Probabilidad"] = df["Probabilidad"].apply(lambda x: f"{x:.2%}")

        # Display the dataframe
        st.table(df)

    if st.button("Reiniciar juego"):
        st.session_state.game_state = {
            "user_responses": [],
            "game_over": False,
            "show_user_data": st.session_state.game_state["show_user_data"],
            "show_model_info": st.session_state.game_state["show_model_info"],
        }
        akinator_core.reset_game()
        st.experimental_rerun()


if __name__ == "__main__":
    main()
