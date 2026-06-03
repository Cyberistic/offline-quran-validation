"""
VerseTracker — Detect verse boundaries in streaming Quran transcription.

As accumulated text grows, matches against QuranDB and emits (surah, ayah)
when a verse is "complete" (match shifts to a different verse or score drops).
Continuation bias resolves duplicate-text ambiguity by preferring the next
sequential verse after the last emission.

Deferred emission (upstream-aligned):
  When accumulated text overflows a matched verse, the verse is held in
  PENDING state. Emission is only confirmed when a NEW (different) verse
  is matched with sufficient score, OR at finalize() with strong evidence.
  This prevents cascading bogus next-verse emissions on long single-verse
  clips where the overflow is just noise (upstream EXPERIMENTS.md
  "v3 still 57 samples where recall is perfect but SeqAcc is zero
  because the tracker emits many verses after the correct one").
"""

from Levenshtein import ratio
from shared.normalizer import normalize_arabic
from shared.quran_db import QuranDB

CONTINUATION_BONUS = 0.15
SCORE_DROP_THRESHOLD = 0.15  # emit when score drops this much below peak
MIN_EMIT_SCORE = 0.3  # minimum score to consider a match real
OVERFLOW_RATIO = 1.15  # if accumulated words > verse words * this, try splitting
STREAMING_MIN_EMIT_SCORE = 0.4
MIN_WORDS_FOR_MATCH = 2
# Pending emission ages out after this many process_delta calls without
# confirmation. Prevents indefinite hold when the next verse is missed.
MAX_PENDING_AGE = 3
# Minimum score for a new (different) verse to confirm a pending emission.
# Acts as a "next verse must be clearly better than nothing" filter.
PENDING_CONFIRM_SCORE = 0.35


class VerseTracker:
    """Track and emit verse detections from streaming text."""

    def __init__(
        self,
        db: QuranDB = None,
        last_emission: tuple[int, int] | None = None,
        streaming_mode: bool = False,
    ):
        self.db = db or QuranDB()
        self._streaming_mode = streaming_mode
        self._min_emit_score = STREAMING_MIN_EMIT_SCORE if streaming_mode else MIN_EMIT_SCORE
        self._accumulated = ""
        self._current_match: dict | None = None  # {surah, ayah, text_clean, score}
        self._peak_score: float = 0.0
        self._emissions: list[dict] = []
        self._last_emitted: tuple[int, int] | None = last_emission
        # Deferred emission state
        self._pending: dict | None = None  # match dict held for confirmation
        self._pending_age: int = 0  # process_delta calls since _pending was set

    def _score_verse(self, text: str, verse_clean: str, surah: int, ayah: int) -> float:
        """Score a candidate verse against text, with prefix-awareness and continuation bias."""
        text_words = text.split()
        verse_words = verse_clean.split()
        n_text = len(text_words)
        n_verse = len(verse_words)

        # Prefix scoring: compare text against the first N words of the verse
        prefix_len = min(n_text, n_verse)
        prefix = " ".join(verse_words[:prefix_len])
        prefix_score = ratio(text, prefix)
        full_score = ratio(text, verse_clean)

        coverage = n_text / max(n_verse, 1)
        if coverage > 0.8:
            raw = 0.3 * prefix_score + 0.7 * full_score
        else:
            raw = 0.7 * prefix_score + 0.3 * full_score

        # Continuation bias: boost the next expected verse
        if self._last_emitted:
            next_v = self.db.get_next_verse(*self._last_emitted)
            if next_v and next_v["surah"] == surah and next_v["ayah"] == ayah:
                raw += CONTINUATION_BONUS

        return raw

    def _find_best_match(self, text: str) -> dict | None:
        """Find the best matching single verse for the given text."""
        if not text.strip():
            return None

        # Streaming mode: require minimum word count
        if self._streaming_mode and len(text.split()) < MIN_WORDS_FOR_MATCH:
            return None

        best = None
        best_score = 0.0

        for v in self.db.verses:
            score = self._score_verse(text, v["text_clean"], v["surah"], v["ayah"])
            if score > best_score:
                best_score = score
                best = {
                    "surah": v["surah"],
                    "ayah": v["ayah"],
                    "text_clean": v["text_clean"],
                    "score": best_score,
                }

        if best and best["score"] >= self._min_emit_score:
            return best
        return None

    def _confirm_pending(self) -> list[dict]:
        """Emit the held _pending verse. Returns [emission] or []."""
        if not self._pending:
            return []
        emission = {
            "surah": self._pending["surah"],
            "ayah": self._pending["ayah"],
            "score": self._pending["score"],
        }
        self._emissions.append(emission)
        self._last_emitted = (emission["surah"], emission["ayah"])
        # Trim accumulator against the confirmed verse
        matched_words = self._pending["text_clean"].split()
        acc_words = self._accumulated.split()
        overlap = min(len(matched_words), len(acc_words))
        self._accumulated = " ".join(acc_words[overlap:])
        self._pending = None
        self._pending_age = 0
        return [emission]

    def _hold_for_confirmation(self, match: dict) -> None:
        """Move a completed verse into _pending instead of emitting immediately."""
        self._pending = match
        self._pending_age = 0
        # Also reset current match (we're now past it)
        self._current_match = None
        self._peak_score = 0.0

    def _emit_immediate(self, match: dict) -> dict | None:
        """Emit a verse without confirmation. Use for same-verse updates only."""
        ref = (match["surah"], match["ayah"])
        if ref == self._last_emitted:
            self._current_match = None
            self._peak_score = 0.0
            return None

        emission = {"surah": match["surah"], "ayah": match["ayah"], "score": match["score"]}
        self._emissions.append(emission)
        self._last_emitted = ref

        matched_words = match["text_clean"].split()
        acc_words = self._accumulated.split()
        overlap = min(len(matched_words), len(acc_words))
        remaining_words = acc_words[overlap:]
        self._accumulated = " ".join(remaining_words)

        self._current_match = None
        self._peak_score = 0.0

        return emission

    def _try_split_and_hold(self, match: dict) -> list[dict]:
        """If accumulated text overflows the matched verse, hold the verse
        for deferred confirmation rather than emitting immediately.

        Returns emissions only if a previously held pending was confirmed
        during this check.
        """
        emissions = []
        acc_words = self._accumulated.split()
        verse_words = match["text_clean"].split()

        if (
            len(acc_words) > len(verse_words) * OVERFLOW_RATIO
            and len(verse_words) > 0
            and match["score"] >= self._min_emit_score
        ):
            # Hold the matched verse for confirmation instead of emitting
            self._hold_for_confirmation(match)

        return emissions

    def _evaluate(self) -> list[dict]:
        """Core evaluation logic — find matches and emit completed verses."""
        emissions = []

        # Age the pending emission each evaluation
        if self._pending is not None:
            self._pending_age += 1
            if self._pending_age > MAX_PENDING_AGE:
                # Held too long without confirmation — emit anyway
                emissions.extend(self._confirm_pending())

        match = self._find_best_match(self._accumulated)
        if not match:
            return emissions

        # If we have a pending and a new (different) verse matched, confirm pending
        if (
            self._pending is not None
            and match["score"] >= PENDING_CONFIRM_SCORE
            and (match["surah"], match["ayah"]) != (self._pending["surah"], self._pending["ayah"])
        ):
            emissions.extend(self._confirm_pending())

        same_verse = (
            self._current_match
            and self._current_match["surah"] == match["surah"]
            and self._current_match["ayah"] == match["ayah"]
        )

        if same_verse:
            if match["score"] > self._peak_score:
                self._peak_score = match["score"]
            elif self._peak_score - match["score"] > SCORE_DROP_THRESHOLD:
                # Score dropped — hold for confirmation instead of immediate emit
                if self._current_match["score"] >= self._min_emit_score:
                    self._hold_for_confirmation(self._current_match)
                else:
                    self._current_match = None
                    self._peak_score = 0.0
            else:
                self._current_match = match
        else:
            # Different verse — current one is implicitly "complete" if it was
            # real (score above threshold). Hold it for confirmation.
            if (
                self._current_match
                and self._current_match["score"] >= self._min_emit_score
                and not self._pending  # don't double-hold
            ):
                self._hold_for_confirmation(self._current_match)
            self._current_match = match
            self._peak_score = match["score"]

        if not self._current_match and not self._pending:
            self._current_match = match
            self._peak_score = match["score"]

        if self._current_match and not emissions:
            split_emissions = self._try_split_and_hold(self._current_match)
            if split_emissions:
                emissions.extend(split_emissions)

        return emissions

    def process_text(self, text: str) -> list[dict]:
        """Process full accumulated text, return any verse emissions.

        Args:
            text: The full accumulated transcript so far (not a delta).
                  Resets internal accumulator to this text.

        Returns:
            List of emitted verses [{"surah": int, "ayah": int, "score": float}]
        """
        normalized = normalize_arabic(text)
        if not normalized.strip():
            return []

        self._accumulated = normalized
        return self._evaluate()

    def process_delta(self, new_text: str) -> list[dict]:
        """Process new text delta (appended to internal accumulator).

        Use this for streaming — each call appends new words rather than
        resetting the accumulator, so text trimming from previous emissions
        is preserved.

        Args:
            new_text: New text to append (just this chunk's transcript).

        Returns:
            List of emitted verses [{"surah": int, "ayah": int, "score": float}]
        """
        normalized = normalize_arabic(new_text)
        if not normalized.strip():
            return []

        if self._accumulated:
            self._accumulated += " " + normalized
        else:
            self._accumulated = normalized

        return self._evaluate()

    def finalize(self) -> list[dict]:
        """Flush any remaining match as a final emission."""
        emissions = []
        # Confirm any held pending
        if self._pending and self._pending["score"] >= self._min_emit_score:
            emissions.extend(self._confirm_pending())
        # Emit current match if strong enough
        elif self._current_match and self._current_match["score"] >= self._min_emit_score:
            e = self._emit_immediate(self._current_match)
            if e:
                emissions.append(e)
        return emissions
