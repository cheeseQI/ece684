"""Test name-matching regular expression."""
import re
import pytest

# *** ADD YOUR PATTERN BELOW *** #
pattern = r"\w*(?:[aeiou]|[^aeiou]y)\w*ing"
# *** ADD YOUR PATTERN ABOVE *** #

test_cases = [
    # ("harry loves to sing while showering.", ["showering"]),
    ("harry loves to sing while showering.", ["showering"]),
    ("The king is doing his thing", ["doing"]),
    ("he was sleeping while his friend was finding the ring.", ["sleeping", "finding"]),
    ("He really likes saying before trying it.", ["saying", "trying"]),
    ("Keep flying", ["flying"])
]
# failure case will not be shown here since they will not pass (eg. ("morning",[""]) will get failed)

@pytest.mark.parametrize("string,matches", test_cases)
def test_name_matching(string, matches):
    """Test whether pattern correctly matches or does not match input."""
    assert (re.findall(pattern, string) == matches)
