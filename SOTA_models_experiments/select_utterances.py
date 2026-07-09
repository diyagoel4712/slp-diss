"""
Select 10 utterances from the CMU ARCTIC corpus that maximise coverage
of phonological features diagnostic for Arabic, Hindi, Vietnamese,
Korean, and Spanish L2 English accents.

Features targeted (from the phonological feature matrix):
  1.  TH-fricatives    /θ ð/           — th, this, the, breathe, weather...
  2.  Retroflex stops  /ʈ ɖ/ proxy     — word-medial /t d/ (butter, body)
  3.  Stop aspiration  /p t k/          — word-initial voiceless stops
  4.  Rhotics          /r/ onset+coda   — red, further, word-final r
  5.  /v/ vs /w/                        — village, wave, over, window
  6.  /l/ dark/clear   post-vocalic     — ball, full, old, milk
  7.  Vowel lax/tense  KIT/FLEECE       — bit/beat, sit/seat
  8.  Vowel TRAP/DRESS                  — bad/bed, man/men
  9.  Diphthongs        PRICE MOUTH GOAT — I, out, go, my, found
 10.  Schwa/reduction  unstressed Vs    — the, of, a, was, for (function words)
 11.  Onset clusters   /str/ /sp/ /sk/  — street, spring, scratch
 12.  Coda clusters    /st/ /nd/ /lk/   — fast, hand, milk, walked
 13.  Final voiced obs  /d/ /b/ /g/     — bad, web, dog
 14.  Tonal F0 proxy   polysyllabic Wd  — polysyllabic words (Vietnamese stress)
 15.  Rhythm / stress   polysyllabic    — words with clear stress contrast

Each sentence is scored for the number of *distinct* features it covers.
The selection algorithm greedily picks sentences that maximise the total
number of *newly covered* features at each step.

Usage:
    python select_utterances.py arctic.data
    python select_utterances.py arctic.data --n 10 --top_candidates 50
"""

import re
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# 1.  Feature detectors
#     Each returns True if the sentence contains a token of the feature.
#     All checks operate on the lower-cased sentence string for simplicity.
# ---------------------------------------------------------------------------

def has_th_fricatives(s):
    """TH sounds /θ ð/ — 'the', 'this', 'that', 'with', 'think', 'breathe'."""
    # Match 'th' not preceded/followed by 's' (avoid 'months' edge cases)
    return bool(re.search(r'\bth', s.lower()))


def has_word_medial_td(s):
    """
    Proxy for retroflex /ʈ ɖ/ target (Hindi) and flapping target (GenAm):
    words where /t/ or /d/ appears between vowels (butter, body, better,
    mother, water, ready). Catches cases where Korean, Spanish, Arabic
    speakers also differ.
    """
    # Pattern: vowel-letter, then t or d, then vowel-letter (intervocalic)
    return bool(re.search(r'[aeiou][td][aeiou]', s.lower()))


def has_voiceless_stop_initial(s):
    """
    /p t k/ in word-initial position — aspiration target.
    'people', 'time', 'came', 'part', 'took', 'carry' etc.
    """
    return bool(re.search(r'\b[ptk][aeiourlwynm]', s.lower()))


def has_rhotics(s):
    """
    /r/ in onset or coda — rhoticity differs across all 5 L1s.
    Cover both initial (ran) and post-vocalic (further, word, her).
    """
    return bool(re.search(r'[aeiou]r\b|r[aeiou]|\br[aeiou]', s.lower()))


def has_v_w_contrast(s):
    """
    Both /v/ and /w/ in the same sentence, or /v/ alone (most diagnostic).
    Targets Hindi /ʋ/ merger, Spanish /b-v/ merger, Vietnamese /v/→[j/b].
    """
    s_lower = s.lower()
    has_v = bool(re.search(r'\bv|\bov|\bev|ive\b|ove\b|ave\b', s_lower))
    has_w = bool(re.search(r'\bw[aeiou]', s_lower))
    return has_v and has_w


def has_dark_l(s):
    """
    Post-vocalic /l/ — dark-L and L-vocalisation target.
    'ball', 'full', 'old', 'cold', 'milk', 'held', 'fell'.
    """
    return bool(re.search(r'[aeiou]l[^aeiou]|[aeiou]l\b', s.lower()))


def has_kit_fleece(s):
    """
    Both KIT and FLEECE vocabulary in the same sentence, or just KIT words.
    KIT: 'it', 'is', 'in', 'him', 'his', 'if', 'with', 'this', 'big'.
    FLEECE: 'he', 'she', 'we', 'see', 'me', 'be', 'feel', 'keep'.
    """
    s_lower = s.lower()
    kit = bool(re.search(r'\b(it|is|in|him|his|if|this|big|bit|sit|will|still|fill|hill)\b',
                          s_lower))
    fleece = bool(re.search(r'\b(he|she|we|see|me|be|feel|keep|deep|feet|need|read|seen)\b',
                             s_lower))
    return kit and fleece


def has_trap_dress(s):
    """
    TRAP/DRESS vowels — conflated by Spanish and Korean speakers.
    TRAP words: 'and', 'at', 'had', 'back', 'that', 'hand', 'man', 'can'.
    DRESS words: 'said', 'bed', 'red', 'head', 'left', 'ten', 'when', 'then'.
    """
    s_lower = s.lower()
    trap  = bool(re.search(r'\b(and|at|had|back|that|hand|man|can|have|last|fast|past|black|ran)\b',
                            s_lower))
    dress = bool(re.search(r'\b(said|bed|red|head|left|ten|when|then|went|never|yet|get|set|let)\b',
                            s_lower))
    return trap and dress


def has_diphthongs(s):
    """
    PRICE (/aɪ/), MOUTH (/aʊ/), GOAT (/oʊ/) — monophthongised by most L2 groups.
    """
    s_lower = s.lower()
    price = bool(re.search(r'\b(i|my|by|time|like|life|might|night|write|find|mind|kind|side|night|light|right)\b',
                            s_lower))
    mouth = bool(re.search(r'\b(out|now|how|about|found|down|town|around|sound|ground|house)\b',
                            s_lower))
    goat  = bool(re.search(r'\b(go|so|no|know|home|over|old|cold|hold|most|both|only|open|close)\b',
                            s_lower))
    return price or mouth or goat


def has_schwa_contexts(s):
    """
    High density of function words likely to undergo schwa reduction:
    'the', 'a', 'of', 'to', 'for', 'was', 'and', 'in', 'from', 'with'.
    Require at least 3 such tokens for a high-reduction sentence.
    """
    s_lower = s.lower()
    fn_words = re.findall(
        r'\b(the|a|of|to|for|was|and|in|from|with|at|on|is|it|his|her|by)\b',
        s_lower)
    return len(fn_words) >= 3


def has_onset_clusters(s):
    """
    Complex onset clusters: /str/, /spr/, /skr/, /sp/, /st/, /sk/, /gr/, /tr/, /fr/.
    Broken by epenthesis in Vietnamese and Korean; e-epenthesis in Spanish.
    """
    return bool(re.search(
        r'\b(str|spr|skr|spl|scr|squ|shr|sn[aeiou]|sm[aeiou]|sl[aeiou]|sp[aeiou]|st[aeiou]|sk[aeiou]|gr[aeiou]|tr[aeiou]|fr[aeiou]|cr[aeiou]|pr[aeiou]|bl[aeiou]|cl[aeiou]|fl[aeiou]|gl[aeiou]|pl[aeiou])',
        s.lower()))


def has_coda_clusters(s):
    """
    Complex codas: /st/, /nd/, /lk/, /sk/, /kt/, /ft/, /nts/, /sts/.
    Simplified by all five L1 groups in different ways.
    """
    return bool(re.search(
        r'(st|nd|lk|sk|kt|ft|nts|sts|lts|nds|rds|lps|mps|nks)\b',
        s.lower()))


def has_final_voiced_obstruents(s):
    """
    Words ending in /b d g v z/ — devoiced by Arabic and Korean speakers.
    """
    return bool(re.search(
        r'[aeiourlmn][bdgvz]\b',
        s.lower()))


def has_polysyllabic_words(s):
    """
    Polysyllabic words with clear stress contrasts — diagnostic for
    syllable-timing (Hindi, Vietnamese, Spanish, Korean), and F0 tone
    transfer (Vietnamese). Proxy: words of 3+ syllables.
    """
    words = re.findall(r'\b[a-zA-Z]+\b', s)
    long_words = [w for w in words if count_syllables(w) >= 3]
    return len(long_words) >= 2


def count_syllables(word):
    """Simple vowel-cluster syllable counter."""
    return len(re.findall(r'[aeiouAEIOU]+', word))


def has_stress_minimal_pairs(s):
    """
    Noun/verb stress-shift words or clear stress contrast:
    'record', 'present', 'object', 'permit', 'protest', 'produce',
    or simply long words where stress placement differs cross-linguistically.
    """
    stress_words = re.compile(
        r'\b(record|present|object|permit|protest|produce|'
        r'increase|decrease|progress|project|conduct|'
        r'certainly|several|remember|together|important|'
        r'following|particular|different|everything|something)\b',
        re.IGNORECASE)
    return bool(stress_words.search(s))


# ---------------------------------------------------------------------------
# 2.  Feature registry
# ---------------------------------------------------------------------------

FEATURES = [
    ("TH-fricatives /θ ð/",          has_th_fricatives),
    ("Intervocalic /t d/ (retroflex)", has_word_medial_td),
    ("Initial /p t k/ aspiration",    has_voiceless_stop_initial),
    ("Rhotics /r/ onset+coda",        has_rhotics),
    ("/v/ vs /w/ contrast",           has_v_w_contrast),
    ("Dark /l/ post-vocalic",         has_dark_l),
    ("KIT+FLEECE lax/tense vowels",   has_kit_fleece),
    ("TRAP+DRESS vowel pair",         has_trap_dress),
    ("Diphthongs PRICE/MOUTH/GOAT",   has_diphthongs),
    ("Schwa / vowel reduction",       has_schwa_contexts),
    ("Onset consonant clusters",      has_onset_clusters),
    ("Coda consonant clusters",       has_coda_clusters),
    ("Final voiced obstruents",       has_final_voiced_obstruents),
    ("Polysyllabic words (3+ syl)",   has_polysyllabic_words),
    ("Lexical stress contrast words", has_stress_minimal_pairs),
]

FEATURE_NAMES = [name for name, _ in FEATURES]
FEATURE_FUNCS = [fn   for _, fn in FEATURES]
N_FEATURES    = len(FEATURES)


# ---------------------------------------------------------------------------
# 3.  Parse ARCTIC data file
# ---------------------------------------------------------------------------

def parse_arctic(path: str) -> list[tuple[str, str]]:
    """
    Parse lines of the form:  ( arctic_a0001 "sentence text here." )
    Returns list of (utterance_id, sentence_text).
    """
    pattern = re.compile(r'\(\s*(\S+)\s+"([^"]+)"\s*\)')
    utterances = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                uid  = m.group(1)
                text = m.group(2).strip()
                utterances.append((uid, text))
    return utterances


# ---------------------------------------------------------------------------
# 4.  Score each utterance
# ---------------------------------------------------------------------------

def score_utterance(text: str) -> list[bool]:
    """Return binary feature vector for a sentence."""
    return [fn(text) for fn in FEATURE_FUNCS]


# ---------------------------------------------------------------------------
# 5.  Greedy maximum-coverage selection
# ---------------------------------------------------------------------------

def greedy_select(
    utterances: list[tuple[str, str]],
    n: int = 10,
    top_candidates: int | None = None,
) -> list[tuple[str, str, list[bool]]]:
    """
    Greedy set-cover selection.

    At each step, pick the utterance that covers the most features
    not yet covered by the already-selected set.

    Args:
        utterances:      list of (id, text)
        n:               number of utterances to select
        top_candidates:  if set, pre-filter to top-k by total feature count
                         before greedy selection (faster on large corpora)
    Returns:
        list of (id, text, feature_vector) in selection order
    """
    scored = [(uid, text, score_utterance(text)) for uid, text in utterances]

    # Optional pre-filter: keep only utterances that score at least one feature
    scored = [(uid, text, fv) for uid, text, fv in scored if any(fv)]

    if top_candidates:
        scored.sort(key=lambda x: sum(x[2]), reverse=True)
        scored = scored[:top_candidates]

    covered    = [False] * N_FEATURES
    selected   = []
    remaining  = list(scored)

    for _ in range(n):
        if not remaining:
            break

        # Pick utterance with most newly covered features
        best_idx   = -1
        best_gain  = -1
        best_entry = None

        for i, (uid, text, fv) in enumerate(remaining):
            gain = sum(1 for j, v in enumerate(fv) if v and not covered[j])
            if gain > best_gain:
                best_gain  = gain
                best_idx   = i
                best_entry = (uid, text, fv)

        if best_entry is None or best_gain == 0:
            # No more new features to cover; pick highest-scoring remaining
            remaining.sort(key=lambda x: sum(x[2]), reverse=True)
            best_entry = remaining[0]
            best_idx   = 0

        selected.append(best_entry)
        uid, text, fv = best_entry

        # Update covered set
        for j, v in enumerate(fv):
            if v:
                covered[j] = True

        remaining.pop(best_idx)

    return selected


# ---------------------------------------------------------------------------
# 6.  Report
# ---------------------------------------------------------------------------

def print_report(selected: list[tuple[str, str, list[bool]]]) -> None:
    total_covered = [False] * N_FEATURES

    print("\n" + "=" * 70)
    print("SELECTED UTTERANCES — PHONOLOGICAL FEATURE COVERAGE")
    print("=" * 70)

    for rank, (uid, text, fv) in enumerate(selected, 1):
        covered_here = [FEATURE_NAMES[j] for j, v in enumerate(fv) if v]
        new_here     = [FEATURE_NAMES[j] for j, v in enumerate(fv)
                        if v and not total_covered[j]]
        for j, v in enumerate(fv):
            if v:
                total_covered[j] = True

        print(f"\n{'─'*70}")
        print(f"  #{rank:02d}  [{uid}]")
        print(f"        \"{text}\"")
        print(f"  Features covered ({len(covered_here)}): "
              f"{', '.join(covered_here) or 'none'}")
        if new_here:
            print(f"  New features (+{len(new_here)}):  "
                  f"{', '.join(new_here)}")

    print(f"\n{'='*70}")
    print("COVERAGE SUMMARY")
    print(f"{'='*70}")
    n_covered = sum(total_covered)
    print(f"  Total features covered: {n_covered} / {N_FEATURES}")
    print()
    for j, name in enumerate(FEATURE_NAMES):
        tick = "✓" if total_covered[j] else "✗"
        print(f"  {tick}  {name}")

    print(f"\n{'='*70}")
    print("BARE UTTERANCE LIST (copy-paste ready)")
    print(f"{'='*70}")
    for rank, (uid, text, _) in enumerate(selected, 1):
        print(f"  {rank:02d}. [{uid}]  {text}")
    print()


# ---------------------------------------------------------------------------
# 7.  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Select ARCTIC utterances for L2 accent TTS evaluation."
    )
    parser.add_argument("data_file", help="Path to arctic.data file")
    parser.add_argument("--n",              type=int, default=10,
                        help="Number of utterances to select (default: 10)")
    parser.add_argument("--top_candidates", type=int, default=None,
                        help="Pre-filter to top-k utterances before greedy "
                             "selection (default: use all)")
    args = parser.parse_args()

    utterances = parse_arctic(args.data_file)
    print(f"Parsed {len(utterances)} utterances from {args.data_file}")
    print(f"Selecting {args.n} utterances covering {N_FEATURES} features...")

    selected = greedy_select(utterances, n=args.n,
                             top_candidates=args.top_candidates)
    print_report(selected)


if __name__ == "__main__":
    main()
