import numpy as np
import pytest

from core import AkinatorBNCore
from core.akinator_bayesian import Answer


@pytest.fixture
def akinator():
    csv_file = "tests/data/personajes.csv"
    return AkinatorBNCore(csv_file, debug=True)


def test_initialization(akinator):
    assert akinator.model is not None
    assert akinator.inference is not None
    assert len(akinator.characters) == 5
    assert len(akinator.features) == 4
    assert "Aparece en películas" not in akinator.features


def test_get_next_question(akinator):
    question = akinator.get_next_question()
    assert question is not None
    assert question.startswith("¿El personaje ")
    assert question.endswith("?")


def test_process_answer(akinator):
    akinator.get_next_question()
    akinator.process_answer(Answer.YES)
    assert len(akinator.evidence) == 1
    assert list(akinator.evidence.values())[0] == Answer.YES

    akinator.get_next_question()
    akinator.process_answer(Answer.NO)
    assert len(akinator.evidence) == 2
    assert list(akinator.evidence.values())[1] == Answer.NO

    akinator.get_next_question()
    akinator.process_answer(Answer.UNCERTAIN)
    assert len(akinator.evidence) == 3
    assert list(akinator.evidence.values())[2] == Answer.UNCERTAIN


def test_make_guess(akinator):
    akinator.evidence = {
        "Es humano": Answer.YES,
        "Es ficticio": Answer.YES,
        "Tiene poderes mágicos": Answer.YES,
        "Es un hombre": Answer.YES,
    }
    guess, probability = akinator.make_guess()
    assert guess == "Harry Potter"
    assert 0.9 < probability <= 1  # More specific probability range


def test_get_top_guesses(akinator):
    akinator.evidence = {
        "Es humano": Answer.YES,
        "Es ficticio": Answer.YES,
        "Es un hombre": Answer.YES,
    }
    top_guesses = akinator.get_top_guesses(3)
    assert len(top_guesses) == 3
    assert all(isinstance(char, str) and 0 <= prob <= 1 for char, prob in top_guesses)
    assert top_guesses[0][0] in ["Harry Potter", "Batman"]  # Most likely characters


def test_should_stop(akinator):
    akinator.evidence = {
        "Es humano": Answer.YES,
        "Es ficticio": Answer.YES,
        "Tiene poderes mágicos": Answer.YES,
        "Es un hombre": Answer.YES,
    }
    assert akinator.should_stop()

    akinator.evidence = {"Es humano": Answer.YES}
    assert not akinator.should_stop()


def test_reset_game(akinator):
    akinator.evidence = {"Es humano": Answer.YES}
    akinator.current_feature = "Es humano"
    akinator.reset_game()
    assert akinator.evidence == {}
    assert akinator.current_feature is None


def test_select_best_feature(akinator):
    best_feature = akinator.select_best_feature({})
    assert best_feature in akinator.features

    # Test with some evidence
    akinator.evidence = {"Es humano": Answer.YES}
    best_feature = akinator.select_best_feature(akinator.evidence)
    assert best_feature in akinator.features
    assert best_feature != "Es humano"


def test_get_character_data(akinator):
    character_data = akinator.get_character_data("Harry Potter")
    assert character_data["Es humano"] == Answer.YES.value
    assert character_data["Es ficticio"] == Answer.YES.value
    assert character_data["Tiene poderes mágicos"] == Answer.YES.value
    assert character_data["Es un hombre"] == Answer.YES.value

    character_data = akinator.get_character_data("Pikachu")
    assert character_data["Es humano"] == Answer.NO.value
    assert character_data["Es ficticio"] == Answer.YES.value
    assert character_data["Tiene poderes mágicos"] == Answer.YES.value
    assert character_data["Es un hombre"] == Answer.NO.value


def test_max_questions_limit(akinator):
    akinator.max_questions = 2
    for _ in range(akinator.max_questions):
        question = akinator.get_next_question()
        assert question is not None
        akinator.process_answer(Answer.YES)

    assert akinator.get_next_question() is None


def test_update_probabilities(akinator):
    evidence = (("Es humano", Answer.YES), ("Es ficticio", Answer.YES))
    probs = akinator.update_probabilities(evidence)
    assert len(probs) == 5  # All characters
    assert sum(prob for _, prob in probs) == pytest.approx(1.0)
    assert probs[0][0] in ["Harry Potter", "Batman"]  # Most likely characters


def test_calculate_information_gain(akinator):
    evidence = (("Es humano", Answer.YES),)
    info_gain = akinator.calculate_information_gain("Es ficticio", evidence)
    assert 0 <= info_gain <= 1


def test_confidence_threshold(akinator):
    akinator.confidence_threshold = 0.9

    # Test with partial evidence (should not stop)
    akinator.evidence = {
        "Es humano": Answer.YES,
    }
    assert not akinator.should_stop()

    # Test with more evidence (should stop for Harry Potter)
    akinator.evidence = {
        "Es humano": Answer.YES,
        "Tiene poderes mágicos": Answer.YES,
    }
    assert akinator.should_stop()

    # Test with conflicting evidence (should not stop)
    akinator.evidence = {
        "Es humano": Answer.YES,
        "Tiene poderes mágicos": Answer.NO,
    }
    assert not akinator.should_stop()


def test_invalid_character(akinator):
    with pytest.raises(
        KeyError, match="Character 'Invalid Character' not found in the dataset"
    ):
        akinator.get_character_data("Invalid Character")


def test_make_guess_confidence(akinator):
    akinator.evidence = {
        "Es humano": Answer.YES,
        "Tiene poderes mágicos": Answer.YES,
    }
    guess, probability = akinator.make_guess()
    assert guess == "Harry Potter"
    assert np.isclose(probability, 1.0, atol=1e-6)


def test_all_evidence_uncertain(akinator):
    akinator.evidence = {
        "Es humano": Answer.UNCERTAIN,
        "Es ficticio": Answer.UNCERTAIN,
        "Tiene poderes mágicos": Answer.UNCERTAIN,
        "Es un hombre": Answer.UNCERTAIN,
    }

    # Get top guesses for all characters
    top_guesses = akinator.get_top_guesses(len(akinator.characters))

    # Check that we have probabilities for all characters
    assert len(top_guesses) == len(akinator.characters)

    # Check that all probabilities are greater than 0 and less than 1
    assert all(0 < prob < 1 for _, prob in top_guesses)

    # Check that no character has a significantly higher probability than others
    probabilities = [prob for _, prob in top_guesses]
    assert max(probabilities) < 2 * min(
        probabilities
    ), "One character has a significantly higher probability than others"

    # Make a guess and check its probability
    guess, probability = akinator.make_guess()
    assert 0 < probability < 1
    assert guess in akinator.characters


if __name__ == "__main__":
    pytest.main()
