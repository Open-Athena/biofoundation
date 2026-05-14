import functools
import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import bioframe as bf
import numpy as np
import pandas as pd
import torch
from Bio.Seq import Seq
from pyfaidx import Fasta

from biofoundation.model.base import Tokenizer


NUCLEOTIDES = list("ACGT")
INTERVAL_COORDS = ["chrom", "start", "end"]
VARIANT_COORDS = ["chrom", "pos", "ref", "alt"]

_DNA_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}


def _complement_base(base: str) -> str:
    """Complement A/C/G/T; pass any other character through unchanged.

    Non-ACGT inputs (N, IUPAC codes, etc.) round-trip via the unchanged
    branch — downstream DNA tokenizers collapse them to a single unknown
    token regardless, so the exact value returned here is moot.
    """
    return _DNA_COMPLEMENT.get(base, base)


def _maybe_rc(seq: str, pos: int, strand: Literal["+", "-"]) -> tuple[str, int]:
    """If ``strand == "-"``, reverse-complement ``seq`` and map ``pos`` to
    its position in the RC string. Otherwise return inputs unchanged."""
    if strand == "-":
        seq = str(Seq(seq).reverse_complement())  # type: ignore[no-untyped-call]
        pos = len(seq) - 1 - pos
    return seq, pos


class Genome:
    """Random-access FASTA reader backed by :mod:`pyfaidx`.

    Sequences are read on demand from the FASTA file rather than loaded into
    memory upfront, so ``Genome(...)`` is fast to construct and uses
    near-zero baseline memory.

    A samtools-compatible ``.fai`` index is required. For local files, pyfaidx
    creates one next to the FASTA on first open. For remote paths
    (e.g. ``s3://``), the ``.fai`` must already exist alongside the FASTA.

    Gzipped FASTA input must be **bgzipped** (BGZF, with a ``.gzi``
    companion). Plain gzip is not supported — re-compress with ``bgzip``.

    Remote paths (``s3://``, etc.) require the optional ``s3`` extra
    (``pip install -e .[s3]``), which pulls in ``fsspec`` and ``s3fs``.

    Args:
        path: Local filesystem path or fsspec-compatible URL.
        subset_chroms: If given, only chromosomes in this set are exposed.
        storage_options: Forwarded to ``fsspec.open`` for remote paths
            (e.g. ``{"anon": True}`` for public S3 buckets). Ignored for
            local paths.
    """

    def __init__(
        self,
        path: str | Path,
        subset_chroms: set[str] | None = None,
        storage_options: dict[str, Any] | None = None,
    ):
        self._path: str = str(path)
        self._is_remote: bool = urlparse(self._path).scheme not in ("", "file")
        self._storage_options: dict[str, Any] = dict(storage_options or {})

        # Probe once to capture chromosome sizes, then close so no live fd
        # is inherited across fork() into DataLoader workers.
        with self._open_fasta() as fa:
            keys = [k for k in fa.keys() if subset_chroms is None or k in subset_chroms]
            self._chrom_sizes: dict[str, int] = {k: len(fa[k]) for k in keys}

        self._fa: Fasta | None = None
        self._fa_pid: int = -1

    def _open_fasta(self) -> Fasta:
        if self._is_remote:
            import fsspec

            return Fasta(fsspec.open(self._path, **self._storage_options), as_raw=True)
        return Fasta(self._path, as_raw=True)

    def _fasta(self) -> Fasta:
        pid = os.getpid()
        if self._fa is None or self._fa_pid != pid:
            self._fa = self._open_fasta()
            self._fa_pid = pid
        return self._fa

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
        if chrom not in self._chrom_sizes:
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

        seq: str = self._fasta()[chrom][max(start, 0) : min(end, chrom_size)]

        if start < 0:
            seq = "N" * (-start) + seq  # left padding
        if end > chrom_size:
            seq = seq + "N" * (end - chrom_size)  # right padding

        if strand == "-":
            seq = str(Seq(seq).reverse_complement())  # type: ignore[no-untyped-call]
        return seq

    def __getstate__(self) -> dict[str, Any]:
        # Don't pickle the live Fasta handle: its fd is invalid in a spawn worker.
        state = self.__dict__.copy()
        state["_fa"] = None
        state["_fa_pid"] = -1
        return state


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
    strand: Literal["+", "-"] = "+",
) -> tuple[str, int]:
    """Extract a window around a variant position from the genome.

    The forward (``strand="+"``) window splits as
    ``left_flank | REF | right_flank``, with
    ``left_flank = window_size // 2`` bp and
    ``right_flank = window_size - window_size // 2 - 1`` bp. Odd
    ``window_size`` is symmetric (e.g. 127 + 1 + 127 = 255); even
    ``window_size`` puts the extra base in the left flank (e.g. 2 + 1 + 1 = 4).

    With ``strand="-"``, the same genomic interval is returned reverse-
    complemented. The variant moves to index ``window_size - 1 - window_size // 2``
    (equal to the forward index for odd ``window_size``; shifted by 1 for
    even). The base at that index equals ``complement(example["ref"])``.

    For FWD/RC strand averaging, odd ``window_size`` gives symmetric left/right
    context lengths across strands and is the cleanest choice; even sizes are
    supported but the variant's left-context length differs by 1 between strands.

    Args:
        example: Dictionary containing 'chrom', 'pos', 'ref' keys
        genome: Genome object to extract sequence from
        window_size: Size of the window in bp
        strand: ``"+"`` for the forward strand (default), ``"-"`` for the
            reverse-complemented window.

    Returns:
        Tuple of (sequence, position_within_window)
    """
    center_index = example["pos"] - 1  # 1-based to 0-based
    pos = window_size // 2
    start = center_index - pos
    end = start + window_size
    seq = genome(example["chrom"], start, end, strand=strand).upper()
    assert len(seq) == window_size
    if strand == "-":
        pos = window_size - 1 - pos
        assert seq[pos] == _complement_base(example["ref"])
    else:
        assert seq[pos] == example["ref"]
    return seq, pos


@functools.cache
def _get_special_token_counts(tokenizer: Tokenizer) -> tuple[int, int]:
    """``(n_prefix, n_suffix)`` — special tokens auto-prepended / appended.

    Some tokenizers define ``bos_token_id`` / ``eos_token_id`` but don't
    auto-insert them (HF GPT-2-style); a behavioural probe of ``encode("A")``
    is needed to verify the actual policy.
    """
    try:
        bos_id: int | None = tokenizer.bos_token_id
    except AttributeError:
        bos_id = None
    try:
        eos_id: int | None = tokenizer.eos_token_id
    except AttributeError:
        eos_id = None

    encoded = tokenizer.encode("A")
    n_prefix = 1 if bos_id is not None and encoded[:1] == [bos_id] else 0
    n_suffix = 1 if eos_id is not None and encoded[-1:] == [eos_id] else 0
    return n_prefix, n_suffix


@functools.cache
def _get_nucleotide_token_ids(tokenizer: Tokenizer) -> dict[str, int]:
    """Token IDs for the 4 DNA nucleotides under this tokenizer."""
    n_prefix, _ = _get_special_token_counts(tokenizer)
    return {nuc: tokenizer.encode(nuc)[n_prefix] for nuc in NUCLEOTIDES}


def transform_llr_mlm(
    example: dict[str, Any],
    tokenizer: Tokenizer,
    genome: Genome,
    window_size: int,
    strand: Literal["+", "-"] = "+",
) -> dict[str, Any]:
    """Prepare an example for masked language modeling log likelihood ratio scoring.

    The input dictionary follows VCF semantics where `pos` is a 1-based
    coordinate and `ref`/`alt` are single nucleotides. The function extracts a
    centered window from the provided genome, masks the reference position, and
    returns tokenized tensors along with the reference and alternate token
    encodings.

    With ``strand="-"``, the window is reverse-complemented and the ref/alt
    nucleotides are complemented before the ``nuc_ids`` lookup. The masked
    position is at the corresponding RC-strand DNA index, then shifted by
    any BOS prefix the tokenizer auto-inserts.
    """
    seq, pos = _get_variant_window(example, genome, window_size, strand=strand)
    input_ids = torch.tensor(tokenizer.encode(seq))
    nuc_ids = _get_nucleotide_token_ids(tokenizer)
    n_prefix, _ = _get_special_token_counts(tokenizer)
    tokenized_pos = pos + n_prefix
    input_ids[tokenized_pos] = tokenizer.mask_token_id
    ref = example["ref"] if strand == "+" else _complement_base(example["ref"])
    alt = example["alt"] if strand == "+" else _complement_base(example["alt"])
    return dict(
        input_ids=input_ids,
        pos=tokenized_pos,
        ref=nuc_ids[ref],
        alt=nuc_ids[alt],
    )


def transform_llr_clm(
    example: dict[str, Any],
    tokenizer: Tokenizer,
    genome: Genome,
    window_size: int,
    strand: Literal["+", "-"] = "+",
) -> dict[str, Any]:
    """Prepare an example for causal language modeling log likelihood ratio scoring.

    The input dictionary follows VCF semantics where `pos` is a 1-based
    coordinate and `ref`/`alt` are single nucleotides. The function extracts a
    centered window from the provided genome, creates two sequences (ref and alt),
    and returns tokenized tensors stacked together.

    With ``strand="-"``, the window is reverse-complemented and ``alt`` is
    complemented before substitution; the variant ends up at the RC-strand
    DNA index inside the window.
    """
    seq, pos = _get_variant_window(example, genome, window_size, strand=strand)
    alt = example["alt"] if strand == "+" else _complement_base(example["alt"])
    ref_seq = seq
    alt_seq = seq[:pos] + alt + seq[pos + 1 :]
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
    strand: Literal["+", "-"] = "+",
) -> dict[str, Any]:
    """Transform a sequence example for reference log probability MLM inference.

    This function prepares a sequence example for masked language modeling by:
    1. Tokenizing the input sequence
    2. Masking a specific position in the sequence
    3. Recording the reference token at that position

    With ``strand="-"``, the input sequence is reverse-complemented and the
    position is mapped to ``len(seq) - 1 - pos`` before tokenization. The
    recorded ``ref`` is then automatically the token id of the complemented
    base.

    Args:
        example: Dictionary containing the sequence data. Must have a key matching
            `seq_col` that contains the input sequence.
        tokenizer: Tokenizer for converting text to token IDs.
        strand: ``"+"`` for the forward strand (default), ``"-"`` for the
            reverse-complemented strand.

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
    assert example["seq"][example["pos"]] in NUCLEOTIDES
    seq, pos = _maybe_rc(example["seq"], example["pos"], strand)
    input_ids = torch.tensor(tokenizer.encode(seq))
    n_prefix, _ = _get_special_token_counts(tokenizer)
    tokenized_pos = pos + n_prefix
    ref = input_ids[tokenized_pos].item()
    input_ids[tokenized_pos] = tokenizer.mask_token_id
    return dict(input_ids=input_ids, pos=tokenized_pos, ref=ref)


def transform_reflogprob_clm(
    example: dict[str, Any],
    tokenizer: Tokenizer,
    strand: Literal["+", "-"] = "+",
) -> dict[str, Any]:
    """Transform a sequence example for reference log probability CLM inference.

    Produces a ``[4, L]`` tensor with all four nucleotides substituted at the
    specified position. The ``ref`` field is the index into ``NUCLEOTIDES``
    (``A``/``C``/``G``/``T``) of the reference base at that position.

    With ``strand="-"``, the input sequence is reverse-complemented and the
    position is mapped to ``len(seq) - 1 - pos``. ``ref`` is then the index
    of the complemented base (because ``seq[pos]`` after RC is the complement
    of the original) — no extra complementation logic needed at the lookup.
    """
    assert example["seq"][example["pos"]] in NUCLEOTIDES
    seq, pos = _maybe_rc(example["seq"], example["pos"], strand)
    input_ids = torch.tensor(tokenizer.encode(seq))
    nuc_ids = _get_nucleotide_token_ids(tokenizer)
    n_prefix, _ = _get_special_token_counts(tokenizer)
    tokenized_pos = pos + n_prefix
    new_input_ids = input_ids.unsqueeze(0).repeat(len(NUCLEOTIDES), 1)
    for i, nuc in enumerate(NUCLEOTIDES):
        new_input_ids[i, tokenized_pos] = nuc_ids[nuc]
    ref = NUCLEOTIDES.index(seq[pos])
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

    n_prefix, n_suffix = _get_special_token_counts(tokenizer)
    body_len = len(full_ids) - n_prefix - n_suffix
    assert body_len == len(seq), (
        "Char-level tokenization required for case-breakdown LL "
        f"(body={body_len} tokens vs {len(seq)} chars; "
        "either non-char-level or unexpected special tokens)."
    )

    is_upper = [False] * n_prefix + [c.isupper() for c in seq] + [False] * n_suffix
    return dict(
        input_ids=torch.tensor(full_ids, dtype=torch.long),
        is_upper=torch.tensor(is_upper, dtype=torch.bool),
    )
