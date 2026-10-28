from __future__ import annotations

from validate_package import validate_keyword_input_format, validate_required_card_layouts


def must_fail(deck_text: str, expected_fragment: str) -> None:
    try:
        validate_keyword_input_format(deck_text)
    except AssertionError as exc:
        assert expected_fragment in str(exc), exc
    else:
        raise AssertionError("keyword-format validation unexpectedly passed")


def main() -> None:
    long_numeric_item = "2.000000000e-02"
    standard_deck = f"*KEYWORD\n*CONTROL_TERMINATION\n{long_numeric_item},0\n*END\n"
    must_fail(standard_deck, "LONG=Y is absent")

    long_deck = standard_deck.replace("*KEYWORD", "*KEYWORD LONG=Y", 1)
    result = validate_keyword_input_format(long_deck)
    assert result["mode"] == "long"
    assert result["maximum_numeric_item_chars"] == len(long_numeric_item)

    oversized_item = "1.23456789012345678901"
    oversized_deck = f"*KEYWORD LONG=Y\n*CONTROL_TERMINATION\n{oversized_item}\n*END\n"
    must_fail(oversized_deck, "20-character field width")

    malformed_layout = """*KEYWORD LONG=Y
*CONTROL_CONTACT
0,0,2,0,1,0,1
*DATABASE_BINARY_D3PLOT
5.0e-4
*CONTACT_ERODING_SINGLE_SURFACE
0,0,0,0,0,0,0,0
0,0,0,0,20,0,0,0
0,0,0,0,0,0,0,0
0,0,1
*INITIAL_VELOCITY_GENERATION
2001,2,0,0,0,-1200
0,0,0,0,0,1,0
*END
"""
    try:
        validate_required_card_layouts(malformed_layout)
    except AssertionError as exc:
        assert "two required R15 cards" in str(exc), exc
    else:
        raise AssertionError("malformed required-card layout unexpectedly passed")

    complete_layout = """*KEYWORD LONG=Y
*CONTROL_CONTACT
0,0,2,0,1,0,1,0
0,0,0,0,0,0,0,0
*DATABASE_BINARY_D3PLOT
5.0e-4
*CONTACT_ERODING_SINGLE_SURFACE
0,0,0,0,0,0,0,0
0,0,0,0,20,0,0,0
0,0,0,0,0,0,0,0
1,1,1
2,0,0,0,3,5,0,0
*INITIAL_VELOCITY_GENERATION
2001,2,0,0,0,-1200,0,0
0,0,0,0,0,1,0,0
*END
"""
    layout_result = validate_required_card_layouts(complete_layout)
    assert layout_result["control_contact_fields_per_card"] == [8, 8]
    assert layout_result["initial_velocity_fields_per_card"] == [8, 8]

    wrong_phase_type = complete_layout.replace(
        "0,0,0,0,0,1,0,0\n*END", "0,0,0,0,0,1,0.000000000e+00,0\n*END"
    )
    try:
        validate_required_card_layouts(wrong_phase_type)
    except AssertionError as exc:
        assert "PHASE/IRIGID requires integer serialization" in str(exc), exc
    else:
        raise AssertionError("floating-point PHASE unexpectedly passed")

    wrong_ecdt_type = complete_layout.replace(
        "0,0,0,0,0,0,0,0\n*DATABASE_BINARY_D3PLOT",
        "0,0,0,0,0,0,0.000000000e+00,0\n*DATABASE_BINARY_D3PLOT",
    )
    try:
        validate_required_card_layouts(wrong_ecdt_type)
    except AssertionError as exc:
        assert "*CONTROL_CONTACT Card 2 requires integer serialization" in str(exc), exc
    else:
        raise AssertionError("floating-point ECDT unexpectedly passed")
    print("keyword-format self-test: PASS")


if __name__ == "__main__":
    main()
