import pathlib

from qp_head_to_head.render import preregistration_tables

DOC = pathlib.Path(__file__).resolve().parents[3] / "docs" / "benchmarking" / "qp-head-to-head.md"
BEGIN = "<!-- preregistration:begin -->"
END = "<!-- preregistration:end -->"


def frozen_block() -> str:
    text = DOC.read_text(encoding="utf-8")
    return text[text.index(BEGIN) + len(BEGIN):text.index(END)]


def test_the_document_has_exactly_one_block_in_order():
    text = DOC.read_text(encoding="utf-8")
    assert text.count(BEGIN) == 1 and text.count(END) == 1
    assert text.index(BEGIN) < text.index(END)


def test_the_block_contains_the_rendered_tables_verbatim():
    assert preregistration_tables() in frozen_block()


def test_the_block_states_the_three_rules():
    block = frozen_block()
    for heading in ("### Rule A: block-structured storage", "### Rule B: dense default per regime and tier",
                    "### Rule C: replacing OSQP in ctrlpp's linear MPC"):
        assert heading in block
