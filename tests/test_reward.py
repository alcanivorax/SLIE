from slie.models import SLIEAction
from slie.reward import compute_reward


def test_reward_perfect_non_final() -> None:
    action = SLIEAction(intent="greeting", confidence=0.9, response="hello and welcome")
    step_spec = {
        "expected_intent": "greeting",
        "intent_aliases": ["hello"],
        "expected_keywords": ["hello"],
    }
    reward, _ = compute_reward(action, step_spec, None, is_final_step=False)
    assert reward == 0.7


def test_reward_final_clamped() -> None:
    action = SLIEAction(intent="greeting", confidence=0.9, response="hello")
    step_spec = {
        "expected_intent": "greeting",
        "intent_aliases": ["hello"],
        "expected_keywords": ["hello"],
    }
    reward, _ = compute_reward(action, step_spec, None, is_final_step=True)
    assert reward == 0.99
