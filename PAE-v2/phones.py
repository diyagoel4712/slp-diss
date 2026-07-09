"""
utils/phones.py — ARPAbet phone inventory, phonological categories,
and diagnostic phone sets for accent discrimination.

The key insight motivating this module:
  Not all phones are equally informative about accent.
  The TRAP-BATH split is only observable from /æ/ and /ɑː/ tokens.
  Rhoticity is only observable from /ɛr/, /ɪr/, /ʊr/ and post-vocalic /r/.
  We want to pool *selectively* over diagnostically-relevant phones.

Reference: Wells (1982), Accents of English, Cambridge UP — the standard
  lexical set framework (TRAP, BATH, STRUT, etc.) used in variationist work.
"""

from dataclasses import dataclass, field
from typing import FrozenSet, Dict, List


# ---------------------------------------------------------------------------
# Full ARPAbet inventory
# ---------------------------------------------------------------------------

VOWELS: FrozenSet[str] = frozenset({
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY", "IH", "IY",
    "OW", "OY", "UH", "UW",
})

CONSONANTS: FrozenSet[str] = frozenset({
    "B", "CH", "D", "DH", "F", "G", "HH", "JH",
    "K", "L", "M", "N", "NG", "P", "R", "S",
    "SH", "T", "TH", "V", "W", "Y", "Z", "ZH",
})

STOPS: FrozenSet[str] = frozenset({"P", "T", "K", "B", "D", "G"})
FRICATIVES: FrozenSet[str] = frozenset({"F", "V", "S", "Z", "SH", "ZH", "TH", "DH", "HH"})
NASALS: FrozenSet[str] = frozenset({"M", "N", "NG"})
LIQUIDS: FrozenSet[str] = frozenset({"L", "R"})
GLIDES: FrozenSet[str] = frozenset({"W", "Y"})
AFFRICATES: FrozenSet[str] = frozenset({"CH", "JH"})

ALL_PHONES: FrozenSet[str] = VOWELS | CONSONANTS

# Strip stress digits from ARPAbet labels (e.g. "AE1" → "AE")
def strip_stress(label: str) -> str:
    return label.rstrip("0123456789")


# ---------------------------------------------------------------------------
# Wells lexical set → ARPAbet mapping
# Diagnostic: these phone categories are the loci of major English accent variation
# ---------------------------------------------------------------------------

# Wells keyword → ARPAbet phone(s) that realise it in GA
WELLS_SET: Dict[str, List[str]] = {
    "TRAP":   ["AE"],         # TRAP-BATH split (British varieties raise/lengthen)
    "BATH":   ["AE", "AA"],   # same vowel in American, different in RP
    "STRUT":  ["AH"],         # FOOT-STRUT split (Northern English)
    "FOOT":   ["UH"],
    "LOT":    ["AA"],         # LOT-CLOTH merger (American)
    "CLOTH":  ["AO", "AA"],
    "THOUGHT": ["AO"],        # cot-caught merger in American
    "GOAT":   ["OW"],         # monophthong in many British varieties
    "FACE":   ["EY"],
    "PRICE":  ["AY"],
    "MOUTH":  ["AW"],
    "NEAR":   ["IH", "R"],    # intrinsically rhotic in American
    "SQUARE": ["EH", "R"],
    "START":  ["AA", "R"],    # rhotic vs non-rhotic key environment
    "NORTH":  ["AO", "R"],
    "FORCE":  ["AO", "R"],
    "CURE":   ["UH", "R"],
    "NURSE":  ["ER"],         # rhotic vowel — primary rhoticity indicator
    "COMMA":  ["AH"],
    "LETTER": ["ER", "AH"],   # post-vocalic r environment
}

# ---------------------------------------------------------------------------
# Diagnostic phone sets for accent
# These are the phones we selectively pool over for the accent embedding.
# Chosen because their realisation is most variable across English varieties.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiagnosticPhoneGroup:
    name: str
    phones: FrozenSet[str]
    description: str
    accent_relevance: str

DIAGNOSTIC_GROUPS: List[DiagnosticPhoneGroup] = [
    DiagnosticPhoneGroup(
        name="front_vowels",
        phones=frozenset({"AE", "EH", "IH", "IY"}),
        description="Front vowels",
        accent_relevance="TRAP-BATH split, vowel shift (Northern Cities, Canadian Raising)",
    ),
    DiagnosticPhoneGroup(
        name="back_vowels",
        phones=frozenset({"AA", "AO", "OW", "UH", "UW"}),
        description="Back vowels",
        accent_relevance="Cot-caught merger, LOT-CLOTH, GOAT realisation",
    ),
    DiagnosticPhoneGroup(
        name="rhotic_vowel",
        phones=frozenset({"ER"}),
        description="Rhotacised mid-central vowel (NURSE)",
        accent_relevance="Primary indicator of rhoticity — present in rhotic varieties only",
    ),
    DiagnosticPhoneGroup(
        name="liquid_r",
        phones=frozenset({"R"}),
        description="Approximant /r/",
        accent_relevance="Rhotic vs non-rhotic; retroflex vs approximant realisation",
    ),
    DiagnosticPhoneGroup(
        name="alveolar_stop",
        phones=frozenset({"T", "D"}),
        description="Alveolar stops",
        accent_relevance="Flapping (American), glottalisation (British), affrication",
    ),
    DiagnosticPhoneGroup(
        name="lateral",
        phones=frozenset({"L"}),
        description="Lateral approximant",
        accent_relevance="Dark L vs clear L, vocalisation in some British varieties",
    ),
    DiagnosticPhoneGroup(
        name="diphthongs",
        phones=frozenset({"AY", "AW", "OY", "EY", "OW"}),
        description="Diphthongs",
        accent_relevance="Canadian Raising, monophthongisation (Southern American), smoothing",
    ),
]

# Flat set of all diagnostic phones
DIAGNOSTIC_PHONES: FrozenSet[str] = frozenset(
    p for g in DIAGNOSTIC_GROUPS for p in g.phones
)

# Group name → phones mapping (for pooling)
DIAGNOSTIC_GROUP_MAP: Dict[str, FrozenSet[str]] = {
    g.name: g.phones for g in DIAGNOSTIC_GROUPS
}

# Phone → group name(s) mapping (a phone can belong to multiple groups)
PHONE_TO_GROUPS: Dict[str, List[str]] = {}
for g in DIAGNOSTIC_GROUPS:
    for p in g.phones:
        PHONE_TO_GROUPS.setdefault(p, []).append(g.name)

NUM_DIAGNOSTIC_GROUPS = len(DIAGNOSTIC_GROUPS)


# ---------------------------------------------------------------------------
# Phonological context predicates
# Used to filter tokens to diagnostically-valid contexts
# (e.g. only count /t/ flapping in intervocalic unstressed position)
# ---------------------------------------------------------------------------

def is_intervocalic(prev_phone: str, next_phone: str) -> bool:
    """True if current phone is between two vowel phones."""
    return (strip_stress(prev_phone) in VOWELS and
            strip_stress(next_phone) in VOWELS)


def is_postvocalic_r_context(prev_phone: str, curr_phone: str) -> bool:
    """True if this /R/ token follows a vowel — the rhoticity-diagnostic context."""
    return (strip_stress(curr_phone) == "R" and
            strip_stress(prev_phone) in VOWELS)


def is_word_final(next_phone: str) -> bool:
    """Approximate word-final position (next phone is silence or SIL)."""
    return next_phone in ("SIL", "SP", "", "sil", "sp")


# ---------------------------------------------------------------------------
# Phone inventory index (for contrastive learning phone-category labels)
# ---------------------------------------------------------------------------

# Assign each phone a stable integer ID (for use as contrastive category labels)
PHONE_TO_IDX: Dict[str, int] = {p: i for i, p in enumerate(sorted(ALL_PHONES))}
IDX_TO_PHONE: Dict[int, str] = {i: p for p, i in PHONE_TO_IDX.items()}
NUM_PHONES = len(ALL_PHONES)
