import textwrap

import pytest

from biofoundation.data import Genome


def _write_fasta(tmp_path):
    fasta = textwrap.dedent(
        ">chr1\n"
        "ACGTACGTAC\n"
        ">chr2\n"
        "GGGCCCAGTA\n"
    )
    path = tmp_path / "test.fa"
    path.write_text(fasta)
    return path


def test_genome_returns_subsequence(tmp_path):
    fasta_path = _write_fasta(tmp_path)
    genome = Genome(fasta_path)

    seq = genome("chr1", start=2, end=7)

    assert seq == "GTACG"


def test_genome_reverse_complement(tmp_path):
    fasta_path = _write_fasta(tmp_path)
    genome = Genome(fasta_path)

    seq = genome("chr2", start=1, end=6, strand="-")

    assert seq == "GGGCC"


def test_genome_left_padding(tmp_path):
    fasta_path = _write_fasta(tmp_path)
    genome = Genome(fasta_path)

    seq = genome("chr1", start=-2, end=3)

    assert seq == "NNACG"


def test_genome_right_padding(tmp_path):
    fasta_path = _write_fasta(tmp_path)
    genome = Genome(fasta_path)

    seq = genome("chr1", start=7, end=12)

    assert seq == "TACNN"


def test_genome_padding_both_sides(tmp_path):
    fasta_path = _write_fasta(tmp_path)
    genome = Genome(fasta_path)

    seq = genome("chr1", start=-3, end=12)

    assert seq == "NNNACGTACGTACNN"


def test_genome_requires_known_chromosome(tmp_path):
    fasta_path = _write_fasta(tmp_path)
    genome = Genome(fasta_path)

    with pytest.raises(ValueError, match="chromosome chr3 not found"):
        genome("chr3", start=0, end=1)


def test_genome_validates_span(tmp_path):
    fasta_path = _write_fasta(tmp_path)
    genome = Genome(fasta_path)

    assert genome("chr1", start=5, end=5) == ""
    with pytest.raises(
        ValueError, match="start 6 must be less than or equal to end 4"
    ):
        genome("chr1", start=6, end=4)

    with pytest.raises(ValueError, match="end -1 must be non-negative"):
        genome("chr1", start=-5, end=-1)

    with pytest.raises(ValueError, match="start 11 is out of range"):
        genome("chr1", start=11, end=11)

    with pytest.raises(ValueError, match="start 10 is out of range"):
        genome("chr1", start=10, end=12)


def test_genome_respects_subset(tmp_path):
    fasta_path = _write_fasta(tmp_path)
    genome = Genome(fasta_path, subset_chroms={"chr2"})

    assert genome("chr2", start=0, end=4) == "GGGC"

    with pytest.raises(ValueError, match="chromosome chr1 not found"):
        genome("chr1", start=0, end=4)
