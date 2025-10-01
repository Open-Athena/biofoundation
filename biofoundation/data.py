import gzip
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from transformers import PreTrainedTokenizerBase


NUCLEOTIDES = list("ACGT")


def transform_llr_mlm(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    genome: "Genome",
    window_size: int,
) -> dict[str, Any]:
    """Prepare an example for masked language modeling log likelihood ratio scoring.

    The input dictionary follows VCF semantics where `pos` is a 1-based
    coordinate and `ref`/`alt` are single nucleotides. The function extracts a
    centered window from the provided genome, masks the reference position, and
    returns tokenized tensors along with the reference and alternate token
    encodings.
    """
    center_index = example["pos"] - 1  # 1-based to 0-based
    assert window_size % 2 == 0, "window_size must be even"
    start = center_index - window_size // 2
    end = center_index + window_size // 2
    seq = genome(example["chrom"], start, end).upper()
    assert len(seq) == window_size
    pos = window_size // 2
    assert seq[pos] == example["ref"]
    input_ids = tokenizer(seq, return_tensors="pt")["input_ids"][0]
    input_ids[pos] = tokenizer.mask_token_id
    return dict(
        input_ids=input_ids,
        pos=pos,
        ref=tokenizer(example["ref"])["input_ids"][0],
        alt=tokenizer(example["alt"])["input_ids"][0],
    )


def transform_reflogprob_mlm(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Any]:
    """Transform a sequence example for reference log probability MLM inference.

    This function prepares a sequence example for masked language modeling by:
    1. Tokenizing the input sequence
    2. Masking a specific position in the sequence
    3. Recording the reference token at that position

    Args:
        example: Dictionary containing the sequence data. Must have a key matching
            `seq_col` that contains the input sequence.
        tokenizer: HuggingFace tokenizer for converting text to token IDs.

    Returns:
        Dictionary with three keys:
        - input_ids: Token IDs with the specified position masked
        - pos: The masked position
        - ref: The reference token ID that was at the masked position

    Example:
        >>> example = {"seq": "ATCG"}
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> result = transform_reflogprob_mlm(example, tokenizer, 1)
        >>> print(result)
        {'input_ids': tensor([...]), 'pos': 1, 'ref': 3}
    """
    pos = example["pos"]
    assert example["seq"][pos] in NUCLEOTIDES
    input_ids = tokenizer(example["seq"], return_tensors="pt")["input_ids"][0]
    ref = input_ids[pos].item()
    input_ids[pos] = tokenizer.mask_token_id
    return dict(input_ids=input_ids, pos=pos, ref=ref)


def transform_reflogprob_clm(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Any]:
    pos = example["pos"]
    assert example["seq"][pos] in NUCLEOTIDES
    input_ids = tokenizer(example["seq"], return_tensors="pt")["input_ids"][0]
    ref = input_ids[pos].item()
    # Create 4 copies of the input sequence
    new_input_ids = input_ids.unsqueeze(0).repeat(len(NUCLEOTIDES), 1)
    for i, nuc in enumerate(NUCLEOTIDES):
        new_input_ids[i, pos] = tokenizer.encode(nuc)[0]
    ref = NUCLEOTIDES.index(example["seq"][pos])
    return dict(input_ids=new_input_ids, ref=ref)


def read_fasta(
    path: str | Path,
    subset_chroms: set[str] | None = None,
) -> pd.Series:
    with gzip.open(path, "rt") if str(path).endswith(".gz") else open(path) as handle:
        genome = pd.Series(
            {
                rec.id: str(rec.seq)
                for rec in SeqIO.parse(handle, "fasta")  # type: ignore[no-untyped-call]
                if subset_chroms is None or rec.id in subset_chroms
            }
        )
    return genome


class Genome:
    def __init__(
        self,
        path: str | Path,
        subset_chroms: set[str] | None = None,
    ):
        self._genome: pd.Series = read_fasta(path, subset_chroms=subset_chroms)
        self._chrom_sizes: dict[str, int] = {
            chrom: len(seq) for chrom, seq in self._genome.items()
        }

    def __call__(
        self,
        chrom: str,
        start: int,
        end: int,
        strand: Literal["+", "-"] = "+",
    ) -> str:
        """Get a subsequence of a chromosome.
        If start is negative, the sequence is padded with Ns on the left.
        If end is greater than the chromosome size, the sequence is padded with Ns on
        the right.

        Args:
            chrom: The chromosome to get the sequence of.
            start: The start position of the sequence (0-based, inclusive).
            end: The end position of the sequence (0-based, exclusive).
            strand: The strand of the sequence (+ or -).
        """
        if chrom not in self._genome:
            raise ValueError(f"chromosome {chrom} not found in genome")
        chrom_size = self._chrom_sizes[chrom]
        if strand not in {"+", "-"}:
            raise ValueError("strand must be '+' or '-'")
        if start > end:
            raise ValueError(f"start {start} must be less than or equal to end {end}")
        if end < 0:
            raise ValueError(f"end {end} must be non-negative for chromosome {chrom}")
        if start >= chrom_size:
            raise ValueError(f"start {start} is out of range for chromosome {chrom}")

        seq = self._genome[chrom][max(start, 0) : min(end, chrom_size)]

        if start < 0:
            seq = "N" * (-start) + seq  # left padding
        if end > chrom_size:
            seq = seq + "N" * (end - chrom_size)  # right padding

        if strand == "-":
            seq = str(Seq(seq).reverse_complement())  # type: ignore[no-untyped-call]
        return cast(str, seq)
