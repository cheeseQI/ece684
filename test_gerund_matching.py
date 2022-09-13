"""Test name-matching regular expression."""
import re
import pytest

# *** ADD YOUR PATTERN BELOW *** #
# pattern = r"\b[a-z]*(?:[b-df-hj-np-tv-xz][aieouy][a-z]*)+ing\b"
pattern = r"\w*(?:[aeiou]|[^aeiou]y)\w*ing"
# raise NotImplementedError("Add your pattern to the test file.")
# *** ADD YOUR PATTERN ABOVE *** #

test_cases = [
    # ("harry loves to sing while showering.", ["showering"]),
    ("harry loves to sing while showering.", ["showering"]),
    ("The king is doing his thing", ["doing"]),
    ("he was sleeping while his friend was finding the ring.", ["sleeping", "finding"]),
    ("He really likes saying before trying it.", ["saying", "trying"]),
    ("Keep flying", ["flying"])
]


@pytest.mark.parametrize("string,matches", test_cases)
def test_name_matching(string, matches):
    """Test whether pattern correctly matches or does not match input."""
    assert (re.findall(pattern, string) == matches)
