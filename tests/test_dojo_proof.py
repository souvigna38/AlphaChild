"""Tests for Dojo cryptographic proofs."""

from dojo.proof import make_proof, parse_proof_line, verify_proof


def test_proof_roundtrip():
    secret = "test-secret-at-least-16-chars"
    proof = make_proof("999001", "C2-L12", secret, pytest_target="test_c2_l12_mla.py")
    parsed = parse_proof_line(proof)
    assert parsed is not None
    assert parsed.lesson_id == "C2-L12"
    assert verify_proof("999001", "C2-L12", parsed.token, secret, pytest_target="test_c2_l12_mla.py")
    assert not verify_proof("999002", "C2-L12", parsed.token, secret, pytest_target="test_c2_l12_mla.py")


def test_proof_parse_embedded():
    text = "done! PASS-C1-L01-abcdef012345 thanks"
    parsed = parse_proof_line(text)
    assert parsed is not None
    assert parsed.lesson_id == "C1-L01"
