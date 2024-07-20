import pytest
from core import AkinatorBNCore


@pytest.fixture
def akinator():
    csv_file = "tests/data/personajes.csv"
    return AkinatorBNCore(csv_file, debug=True)


def test_initialization(akinator):
    assert akinator.model is not None
    assert akinator.inference is not None
    assert len(akinator.characters) == 5
    assert (
        len(akinator.features) == 4
    )  # It should ignore `Aparece en películas` since it's always 1


def test_get_next_question(akinator):
    question = akinator.get_next_question()
    assert question is not None
    assert question.startswith("¿El personaje ")
    assert question.endswith("?")


def test_process_answer(akinator):
    akinator.get_next_question()
    akinator.process_answer(True)
    assert len(akinator.evidence) == 1


def test_make_guess(akinator):
    akinator.evidence = {
        "Es humano": 1,
        "Es ficticio": 1,
        "Tiene poderes mágicos": 1,
        "Es un hombre": 1,
    }
    guess, probability = akinator.make_guess()
    assert guess == "Harry Potter"
    assert 0 < probability <= 1


def test_get_top_guesses(akinator):
    akinator.evidence = {"Es humano": 1, "Es ficticio": 1, "Es un hombre": 1}
    top_guesses = akinator.get_top_guesses(3)
    assert len(top_guesses) == 3
    assert all(isinstance(char, str) and 0 <= prob <= 1 for char, prob in top_guesses)


def test_should_stop(akinator):
    akinator.evidence = {
        "Es humano": 1,
        "Es ficticio": 1,
        "Tiene poderes mágicos": 1,
        "Es un hombre": 1,
    }
    assert akinator.should_stop()


def test_reset_game(akinator):
    akinator.evidence = {"Es humano": 1}
    akinator.current_feature = "Es humano"
    akinator.reset_game()
    assert akinator.evidence == {}
    assert akinator.current_feature is None


def test_select_best_feature(akinator):
    best_feature = akinator.select_best_feature({})
    assert best_feature in akinator.features


def test_get_character_data(akinator):
    character_data = akinator.get_character_data("Harry Potter")
    assert character_data["Es humano"] == 1
    assert character_data["Es ficticio"] == 1
    assert character_data["Tiene poderes mágicos"] == 1


def test_max_questions_limit(akinator):
    akinator.max_questions = 2
    for _ in range(akinator.max_questions):
        question = akinator.get_next_question()
        assert question is not None
        akinator.process_answer(True)

    assert akinator.get_next_question() is None


if __name__ == "__main__":
    pytest.main()
