from slie.env import SLIEEnvironment
from slie.models import SLIEAction


def test_env_reset_and_step() -> None:
    env = SLIEEnvironment()
    reset_resp = env.reset("task1", 0)
    assert len(reset_resp.observation.gesture_embedding) == 64
    assert len(reset_resp.observation.hand_landmarks) == 6

    step_resp = env.step(SLIEAction(intent="greeting", confidence=0.9, response="hello"))
    assert step_resp.done is False
    assert step_resp.info.step_count == 1
