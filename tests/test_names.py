from scb.models import Document
from scb.names import deanonymize, deanonymize_all_origins, load_name_groups


def test_load_name_groups():
    groups = load_name_groups()
    assert "turkish" in groups
    assert "swiss_german" in groups
    assert len(groups["turkish"].first_names) > 0


def test_deanonymize_replaces_placeholders():
    doc = Document(id="1", text="A. hat B. am 3. März angegriffen. A. bestritt dies.")
    groups = load_name_groups()
    result = deanonymize(doc, groups["turkish"], seed=0)
    assert "A." not in result.deanonymized_text
    assert "B." not in result.deanonymized_text
    assert len(result.substitutions) == 2


def test_deanonymize_consistent_within_doc():
    doc = Document(id="2", text="A. traf B. A. sprach mit B.")
    groups = load_name_groups()
    result = deanonymize(doc, groups["serbian"], seed=1)
    # A. should always resolve to the same name
    name_a = result.substitutions[0].first_name
    assert result.deanonymized_text.count(name_a) == 2


def test_deanonymize_all_origins():
    doc = Document(id="3", text="X. war angeklagt.")
    variants = deanonymize_all_origins(doc, seed=0)
    origins = {v.name_origin for v in variants}
    assert "turkish" in origins
    assert "swiss_german" in origins
    assert len(variants) == len(origins)
