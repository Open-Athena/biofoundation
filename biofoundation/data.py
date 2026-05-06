import functools
import gzip
from pathlib import Path
from typing import Any, Literal, cast

import bioframe as bf
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq

from biofoundation.model.base import Tokenizer


NUCLEOTIDES = list("ACGT")
INTERVAL_COORDS = ["chrom", "start", "end"]
VARIANT_COORDS = ["chrom", "pos", "ref", "alt"]


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


class GenomicSet:
    """A set of genomic intervals that are always non-overlapping.

    This class represents a collection of genomic intervals (chromosome, start, end)
    with the guarantee that intervals are merged to ensure no overlaps exist within
    the set. The intervals are automatically sorted by chromosome, start, and end
    coordinates. The class supports set-like operations including union (|), intersection (&),
    and subtraction (-).

    Coordinates follow Python semantics:
    - 0-based indexing
    - start is inclusive
    - end is exclusive

    For example, chr1:0-50 represents positions 0 through 49 (50 positions total).

    Note: All intervals are assumed to be unstranded. Strand information is not
    stored or considered in any operations.

    Args:
        data: A pandas DataFrame with columns ['chrom', 'start', 'end']. Any
            overlapping intervals in the input will be merged automatically, and
            the result will be sorted by chromosome, start, and end coordinates.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        if len(data) == 0:
            self._data = pd.DataFrame(columns=INTERVAL_COORDS).astype(
                {"chrom": str, "start": int, "end": int}
            )
        else:
            assert bf.is_bedframe(data, raise_errors=True)
            self._data = bf.merge(data)[INTERVAL_COORDS].sort_values(INTERVAL_COORDS)

    def __repr__(self) -> str:
        return f"GenomicSet\n{self._data}"

    def __or__(self, other: "GenomicSet") -> "GenomicSet":
        """Union of two GenomicSets.

        Returns a new GenomicSet containing all intervals from both sets,
        with overlapping intervals merged.

        Args:
            other: Another GenomicSet to union with.

        Returns:
            A new GenomicSet containing the union of intervals.
        """
        return GenomicSet(pd.concat([self._data, other._data], ignore_index=True))

    def __and__(self, other: "GenomicSet") -> "GenomicSet":
        """Intersection of two GenomicSets.

        Returns a new GenomicSet containing only the overlapping regions
        between the two sets.

        Args:
            other: Another GenomicSet to intersect with.

        Returns:
            A new GenomicSet containing the intersecting intervals.
        """
        return GenomicSet(
            bf.overlap(self._data, other._data, how="inner", return_overlap=True)[
                ["chrom", "overlap_start", "overlap_end"]
            ].rename(columns=dict(overlap_start="start", overlap_end="end"))
        )

    def __sub__(self, other: "GenomicSet") -> "GenomicSet":
        """Subtraction of two GenomicSets.

        Returns a new GenomicSet containing intervals from this set that
        do not overlap with any intervals in the other set.

        Args:
            other: Another GenomicSet to subtract from this set.

        Returns:
            A new GenomicSet containing the remaining intervals.
        """
        return GenomicSet(bf.subtract(self._data, other._data))

    def __eq__(self, other: object) -> bool:
        """Equality comparison based on underlying DataFrame equality.

        Args:
            other: Another GenomicSet to compare with.

        Returns:
            True if both GenomicSets have the same intervals, False otherwise.
        """
        if not isinstance(other, GenomicSet):
            return False
        return bool(self._data.equals(other._data))

    def to_pandas(self) -> pd.DataFrame:
        """Convert the GenomicSet to a pandas DataFrame.

        Returns:
            A pandas DataFrame with columns ['chrom', 'start', 'end']
            containing the non-overlapping intervals, sorted by chromosome,
            start, and end coordinates.
        """
        return self._data

    def n_intervals(self) -> int:
        """Return the number of intervals in the GenomicSet.

        Returns:
            The number of non-overlapping intervals in the set.
        """
        return len(self._data)

    def total_size(self) -> int:
        """Return the total genomic basepairs covered by all intervals.

        Returns:
            The sum of all interval sizes (end - start) in base pairs.
            Since intervals are non-overlapping, this represents the
            actual genomic coverage.
        """
        return int((self._data["end"] - self._data["start"]).sum())

    def expand_min_size(self, min_size: int) -> "GenomicSet":
        """Expand intervals to at least the specified minimum size.

        Each interval is expanded by padding equally on both sides until it
        reaches at least `min_size`. Intervals that are already larger than
        `min_size` are left unchanged.

        Args:
            min_size: Minimum size (in base pairs) for each interval.

        Returns:
            A new GenomicSet with expanded intervals. Overlapping intervals
            resulting from expansion will be automatically merged.
        """
        res = self._data.copy()
        res["size"] = res["end"] - res["start"]
        res["pad"] = np.maximum(
            np.ceil((min_size - res["size"]) / 2).astype(int),
            0,
        )
        res["start"] = res["start"] - res["pad"]
        res["end"] = res["end"] + res["pad"]
        return GenomicSet(res.drop(columns=["size", "pad"]))

    def add_random_shift(self, max_shift: int, seed: int | None = None) -> "GenomicSet":
        """Add random shift to interval positions.

        Each interval is shifted by a random amount (in base pairs) within
        the range [-max_shift, max_shift] (inclusive). The same random shift is applied
        to both start and end positions, preserving the interval size.

        Args:
            max_shift: Maximum absolute shift value in base pairs.
            seed: Random seed for reproducible shifts. If None, shifts will be
                non-reproducible (random each time).

        Returns:
            A new GenomicSet with shifted intervals. Overlapping intervals
            resulting from shifts will be automatically merged.
        """
        rng = np.random.default_rng(seed)
        shift = rng.integers(-max_shift, max_shift, len(self._data), endpoint=True)
        res = self._data.copy()
        res["start"] = res["start"] + shift
        res["end"] = res["end"] + shift
        return GenomicSet(res)


def _get_variant_window(
    example: dict[str, Any],
    genome: "Genome",
    window_size: int,
) -> tuple[str, int]:
    """Extract a window around a variant position from the genome.

    The variant's REF base sits at position ``window_size // 2`` (0-indexed).
    Left flank is ``window_size // 2`` bp; right side is
    ``window_size - window_size // 2`` bp (1 REF + remainder). Even
    ``window_size`` is symmetric; odd ``window_size`` puts the extra base on
    the right (e.g. 127 + 128 for ``window_size=255``).

    Args:
        example: Dictionary containing 'chrom', 'pos', 'ref' keys
        genome: Genome object to extract sequence from
        window_size: Size of the window in bp

    Returns:
        Tuple of (sequence, position_within_window)
    """
    center_index = example["pos"] - 1  # 1-based to 0-based
    left_size = window_size // 2
    start = center_index - left_size
    end = start + window_size
    seq = genome(example["chrom"], start, end).upper()
    assert len(seq) == window_size
    pos = left_size
    assert seq[pos] == example["ref"]
    return seq, pos


@functools.cache
def _get_token_prefix_len(tokenizer: Tokenizer) -> int:
    """Number of special tokens prepended before the first DNA token.

    Inferred by comparing ``encode("A")`` to ``encode("AA")``: both share the
    BOS prefix and EOS suffix; the second has one extra DNA token in between.
    The first index where they diverge is one past the prefix.
    Assumes per-character DNA tokenization.
    """
    e1 = tokenizer.encode("A")
    e2 = tokenizer.encode("AA")
    assert len(e2) == len(e1) + 1, (
        f"tokenizer is not per-character: encode('A')={e1}, encode('AA')={e2}"
    )
    for i in range(len(e1)):
        if e1[i] != e2[i]:
            return i - 1
    return len(e1) - 1


@functools.cache
def _get_nucleotide_token_ids(tokenizer: Tokenizer) -> dict[str, int]:
    """Token IDs for the 4 DNA nucleotides under this tokenizer."""
    prefix_len = _get_token_prefix_len(tokenizer)
    return {nuc: tokenizer.encode(nuc)[prefix_len] for nuc in NUCLEOTIDES}


def transform_llr_mlm(
    example: dict[str, Any],
    tokenizer: Tokenizer,
    genome: Genome,
    window_size: int,
) -> dict[str, Any]:
    """Prepare an example for masked language modeling log likelihood ratio scoring.

    The input dictionary follows VCF semantics where `pos` is a 1-based
    coordinate and `ref`/`alt` are single nucleotides. The function extracts a
    centered window from the provided genome, masks the reference position, and
    returns tokenized tensors along with the reference and alternate token
    encodings.
    """
    seq, pos = _get_variant_window(example, genome, window_size)
    input_ids = torch.tensor(tokenizer.encode(seq))
    nuc_ids = _get_nucleotide_token_ids(tokenizer)
    tokenized_pos = pos + _get_token_prefix_len(tokenizer)
    input_ids[tokenized_pos] = tokenizer.mask_token_id
    return dict(
        input_ids=input_ids,
        pos=tokenized_pos,
        ref=nuc_ids[example["ref"]],
        alt=nuc_ids[example["alt"]],
    )


def transform_llr_clm(
    example: dict[str, Any],
    tokenizer: Tokenizer,
    genome: Genome,
    window_size: int,
) -> dict[str, Any]:
    """Prepare an example for causal language modeling log likelihood ratio scoring.

    The input dictionary follows VCF semantics where `pos` is a 1-based
    coordinate and `ref`/`alt` are single nucleotides. The function extracts a
    centered window from the provided genome, creates two sequences (ref and alt),
    and returns tokenized tensors stacked together.
    """
    seq, pos = _get_variant_window(example, genome, window_size)
    ref_seq = seq
    alt_seq = seq[:pos] + example["alt"] + seq[pos + 1 :]
    input_ids = torch.stack(
        [
            torch.tensor(tokenizer.encode(ref_seq)),
            torch.tensor(tokenizer.encode(alt_seq)),
        ]
    )
    return dict(input_ids=input_ids)


def transform_reflogprob_mlm(
    example: dict[str, Any],
    tokenizer: Tokenizer,
) -> dict[str, Any]:
    """Transform a sequence example for reference log probability MLM inference.

    This function prepares a sequence example for masked language modeling by:
    1. Tokenizing the input sequence
    2. Masking a specific position in the sequence
    3. Recording the reference token at that position

    Args:
        example: Dictionary containing the sequence data. Must have a key matching
            `seq_col` that contains the input sequence.
        tokenizer: Tokenizer for converting text to token IDs.

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
    input_ids = torch.tensor(tokenizer.encode(example["seq"]))
    prefix_len = _get_token_prefix_len(tokenizer)
    tokenized_pos = pos + prefix_len
    ref = input_ids[tokenized_pos].item()
    input_ids[tokenized_pos] = tokenizer.mask_token_id
    return dict(input_ids=input_ids, pos=tokenized_pos, ref=ref)


def transform_reflogprob_clm(
    example: dict[str, Any],
    tokenizer: Tokenizer,
) -> dict[str, Any]:
    pos = example["pos"]
    assert example["seq"][pos] in NUCLEOTIDES
    input_ids = torch.tensor(tokenizer.encode(example["seq"]))
    nuc_ids = _get_nucleotide_token_ids(tokenizer)
    tokenized_pos = pos + _get_token_prefix_len(tokenizer)
    new_input_ids = input_ids.unsqueeze(0).repeat(len(NUCLEOTIDES), 1)
    for i, nuc in enumerate(NUCLEOTIDES):
        new_input_ids[i, tokenized_pos] = nuc_ids[nuc]
    ref = NUCLEOTIDES.index(example["seq"][pos])
    return dict(input_ids=new_input_ids, ref=ref)


def transform_ll_clm(
    example: dict[str, Any],
    tokenizer: Tokenizer,
) -> dict[str, Any]:
    """Prepare an example for CLM sequence-level log-likelihood scoring.

    The raw ``seq`` is uppercased before tokenization, so the tokenizer
    always sees the case it was trained on. The original case is
    preserved in ``is_upper`` for the loss-weight breakdown. This matters
    for case-sensitive (e.g. byte-level) tokenizers such as Evo2's vortex
    CharLevelTokenizer, where ``'a'`` and ``'A'`` map to different token
    ids; for case-insensitive DNA tokenizers like Marin's it is a no-op.

    BOS/EOS handling follows the tokenizer's own policy — we don't
    second-guess it. We detect at most one BOS at the start and one EOS
    at the end (if the tokenizer's ``bos_token_id`` / ``eos_token_id``
    matches there) and mark those positions ``is_upper=False``.
    ``is_upper`` is source-aligned: ``is_upper[i]`` describes the
    character that produced ``input_ids[i]`` (False for special tokens).
    ``compute_ll_clm`` performs the source->target shift when scoring.

    Special-token *targets* (e.g. EOS, when the tokenizer auto-appends
    one) end up with ``is_upper=False`` and so contribute to
    ``ll_sum_lower`` in ``compute_ll_clm`` — a ~1-token bias on
    LL(non-functional) for EOS-trained models, worth knowing if you
    compare absolute LL(non-functional) values across models with vs.
    without EOS.

    Returns:
        input_ids: [L] long tensor.
        is_upper:  [L] bool tensor — True iff ``input_ids[i]`` came from an
                   uppercase character of ``example["seq"]``.
    """
    seq = example["seq"]
    full_ids = tokenizer.encode(seq.upper())

    try:
        bos_id: int | None = tokenizer.bos_token_id
    except AttributeError:
        bos_id = None
    try:
        eos_id: int | None = tokenizer.eos_token_id
    except AttributeError:
        eos_id = None

    n_prefix = 1 if bos_id is not None and full_ids[:1] == [bos_id] else 0
    n_suffix = 1 if eos_id is not None and full_ids[-1:] == [eos_id] else 0
    body_len = len(full_ids) - n_prefix - n_suffix
    assert body_len == len(seq), (
        "Char-level tokenization required for case-breakdown LL "
        f"(body={body_len} tokens vs {len(seq)} chars; "
        "either non-char-level or unexpected special tokens)."
    )

    is_upper = (
        [False] * n_prefix + [c.isupper() for c in seq] + [False] * n_suffix
    )
    return dict(
        input_ids=torch.tensor(full_ids, dtype=torch.long),
        is_upper=torch.tensor(is_upper, dtype=torch.bool),
    )


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
