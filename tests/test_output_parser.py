from humor_detection.output_parser import parse_binary_output


def test_exact_binary_outputs():
    assert parse_binary_output("0").label == 0
    assert parse_binary_output(" 1\n").label == 1
    assert not parse_binary_output("1").parser_failure


def test_ambiguous_outputs_default_to_zero():
    for raw in ["humorous", "Classification: 1", "1.", "", None, "01"]:
        parsed = parse_binary_output(raw)
        assert parsed.label == 0
        assert parsed.parser_failure
