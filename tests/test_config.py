from dataclasses import fields
import ensembles
import pytest


@pytest.mark.parametrize(
    "parameter_set",
    [
        ensembles.config.Parameters,
        ensembles.config.GPRParameters,
        ensembles.config.SGPRParameters,
    ],
)
def test_config(parameter_set: ensembles.config.Parameters):
    c = parameter_set()
    # The dataclass should return a valid dictionary
    c_dict = c.to_dict()
    assert isinstance(c_dict, dict)

    for field in fields(c):
        val = getattr(c, field.name)

        # Let's first check the field was correctly added to the dictionary
        assert field.name in c_dict.keys()
        # And the field's value
        assert c_dict[field.name] == val

        # All parameters should be positive, so let's check that
        assert val > 0

        # The learning rate should be a float, all other parameters are integer-valued
        if field.name in ["learning_rate"]:
            assert isinstance(val, float)
        else:
            assert isinstance(val, int)
