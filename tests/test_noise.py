from slie.env import SLIEEnvironment


def test_observation_noise_is_deterministic_per_seed() -> None:
    env = SLIEEnvironment()

    obs_a = env.reset("task1", 0).observation
    obs_b = env.reset("task1", 0).observation

    assert obs_a.gesture_embedding == obs_b.gesture_embedding
    assert obs_a.hand_landmarks == obs_b.hand_landmarks


def test_observation_noise_changes_with_seed_same_scenario() -> None:
    env = SLIEEnvironment()

    obs_seed0 = env.reset("task1", 0).observation
    obs_seed5 = env.reset("task1", 5).observation  # same scenario id (mod 5), different seed

    assert obs_seed0.detected_gesture == obs_seed5.detected_gesture
    assert (
        obs_seed0.gesture_embedding != obs_seed5.gesture_embedding
        or obs_seed0.hand_landmarks != obs_seed5.hand_landmarks
    )
