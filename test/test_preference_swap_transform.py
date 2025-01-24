from aif_gen.dataset.transforms import PreferenceSwapTransform


def test_apply_preference_swap_to_static_dataset():
    _ = PreferenceSwapTransform()


def test_apply_preference_swap_to_continual_dataset():
    _ = PreferenceSwapTransform()
