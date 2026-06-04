import { ratio } from "./levenshtein";
import { normalizeArabic } from "./normalizer";

const BSM_CLEAN = normalizeArabic("بسم الله الرحمن الرحيم");

export interface QuranVerse {
  surah: number;
  ayah: number;
  text_uthmani: string;
  text_clean: string;
  surah_name: string;
  surah_name_en: string;
}

export interface QuranMatch extends QuranVerse {
  score: number;
}

export interface QuranCandidate {
  surah: number;
  ayah: number;
  ayah_end: number;
  text_clean: string;
  score: number;
}

interface QuranVerseInternal extends QuranVerse {
  text_clean_alt: string;
  text_clean_no_bsm: string | null;
}

interface RunnerUp {
  surah: number;
  ayah: number;
  raw_score: number;
  bonus: number;
  score: number;
  text_clean: string;
}

export interface MatchVerseResult extends QuranCandidate {
  raw_score: number;
  bonus: number;
  runners_up?: RunnerUp[];
}

export class QuranDB {
  private byRef = new Map<string, QuranVerseInternal>();
  private bySurah = new Map<number, QuranVerseInternal[]>();
  private refToIndex = new Map<string, number>();
  private trigramIndex = new Map<string, Uint16Array>();
  private idf = new Map<string, number>();
  public verses: QuranVerseInternal[];

  constructor(verses: QuranVerse[]) {
    this.verses = verses.map((verse) => {
      const textClean = verse.text_clean.replace(/^\uFEFF/, "");
      const textCleanAlt = normalizeArabic(verse.text_uthmani).replace(/^\uFEFF/, "");
      const textCleanNoBsm =
        verse.ayah === 1 && verse.surah !== 1 && verse.surah !== 9 && textClean.startsWith(BSM_CLEAN)
          ? textClean.slice(BSM_CLEAN.length).trim() || null
          : null;
      return {
        ...verse,
        text_clean: textClean,
        text_clean_alt: textCleanAlt,
        text_clean_no_bsm: textCleanNoBsm,
      };
    });

    this.verses.forEach((verse, index) => {
      const ref = `${verse.surah}:${verse.ayah}`;
      this.byRef.set(ref, verse);
      this.refToIndex.set(ref, index);
      const surah = this.bySurah.get(verse.surah) ?? [];
      surah.push(verse);
      this.bySurah.set(verse.surah, surah);
    });

    this.buildTrigramIndex();
  }

  private charTrigrams(text: string): Set<string> {
    if (text.length < 3) return new Set();
    const result = new Set<string>();
    for (let i = 0; i <= text.length - 3; i++) {
      result.add(text.slice(i, i + 3));
    }
    return result;
  }

  private buildTrigramIndex() {
    const posting = new Map<string, Set<number>>();
    const total = this.verses.length;
    this.verses.forEach((verse, index) => {
      const trigrams = this.charTrigrams(verse.text_clean);
      for (const tri of this.charTrigrams(verse.text_clean_alt)) trigrams.add(tri);
      if (verse.text_clean_no_bsm) {
        for (const tri of this.charTrigrams(verse.text_clean_no_bsm)) trigrams.add(tri);
      }
      for (const tri of trigrams) {
        const entry = posting.get(tri) ?? new Set<number>();
        entry.add(index);
        posting.set(tri, entry);
      }
    });

    for (const [tri, indices] of posting) {
      this.trigramIndex.set(tri, Uint16Array.from([...indices].sort((a, b) => a - b)));
      this.idf.set(tri, Math.log(total / indices.size));
    }
  }

  getVerse(surah: number, ayah: number): QuranVerse | undefined {
    return this.byRef.get(`${surah}:${ayah}`);
  }

  getSurah(surah: number): QuranVerse[] {
    return this.bySurah.get(surah) ?? [];
  }

  private continuationBonuses(hint?: [number, number] | null): Map<string, number> {
    const bonuses = new Map<string, number>();
    if (!hint) return bonuses;
    const [surah, ayah] = hint;
    const next = this.getVerse(surah, ayah + 1);
    if (next) {
      bonuses.set(`${surah}:${ayah + 1}`, 0.22);
      if (this.getVerse(surah, ayah + 2)) bonuses.set(`${surah}:${ayah + 2}`, 0.12);
      if (this.getVerse(surah, ayah + 3)) bonuses.set(`${surah}:${ayah + 3}`, 0.06);
    } else {
      const nextSurah = this.getSurah(surah + 1).slice(0, 3);
      [0.22, 0.12, 0.06].forEach((bonus, index) => {
        const verse = nextSurah[index];
        if (verse) bonuses.set(`${verse.surah}:${verse.ayah}`, bonus);
      });
    }
    return bonuses;
  }

  private partialRatio(shortText: string, longText: string): number {
    if (!shortText || !longText) return 0;
    let shortValue = shortText;
    let longValue = longText;
    if (shortValue.length > longValue.length) {
      [shortValue, longValue] = [longValue, shortValue];
    }
    const window = shortValue.length;
    let best = 0;
    for (let i = 0; i < Math.max(1, longValue.length - window + 1); i++) {
      best = Math.max(best, ratio(shortValue, longValue.slice(i, i + window)));
      if (best === 1) break;
    }
    return best;
  }

  private fragmentScore(text: string, verseText: string, fullRatio: number): number {
    const queryWords = text.split(" ");
    const verseWords = verseText.split(" ");
    if (queryWords.length >= 3 && ` ${verseText} `.includes(` ${text} `)) {
      return Math.max(fullRatio, 0.98);
    }
    if (queryWords.length < 4 || verseWords.length < 2) return fullRatio;
    const fragment = this.partialRatio(text, verseText);
    if (fragment <= fullRatio) return fullRatio;
    const shorterPenalty = Math.min(1, verseWords.length / Math.max(queryWords.length, 1));
    const blended = 0.25 * fullRatio + 0.75 * fragment * shorterPenalty;
    return Math.max(fullRatio, blended);
  }

  private bestTextRatio(text: string, verse: QuranVerseInternal): number {
    return Math.max(ratio(text, verse.text_clean), ratio(text, verse.text_clean_alt));
  }

  private bestFragmentScore(text: string, verse: QuranVerseInternal): number {
    return Math.max(
      this.fragmentScore(text, verse.text_clean, ratio(text, verse.text_clean)),
      this.fragmentScore(text, verse.text_clean_alt, ratio(text, verse.text_clean_alt)),
    );
  }

  private suffixPrefixScore(text: string, verseText: string): number {
    const wordsT = text.split(" ");
    const wordsV = verseText.split(" ");
    if (wordsT.length < 2 || wordsV.length < 2) return 0;
    let best = 0;
    const maxTrim = Math.min(Math.floor(wordsT.length / 2), 4);
    for (let trim = 1; trim <= maxTrim; trim++) {
      const suffix = wordsT.slice(trim).join(" ");
      const n = wordsT.length - trim;
      const prefix = wordsV.slice(0, Math.min(n, wordsV.length)).join(" ");
      best = Math.max(best, ratio(suffix, prefix));
    }
    return best;
  }

  trigramCandidates(text: string, topK = 50): number[] {
    const trigrams = this.charTrigrams(text);
    if (trigrams.size === 0) return [];
    const scores = new Map<number, number>();
    for (const tri of trigrams) {
      const weight = this.idf.get(tri);
      const posting = this.trigramIndex.get(tri);
      if (!weight || !posting) continue;
      for (const index of posting) {
        scores.set(index, (scores.get(index) ?? 0) + weight);
      }
    }
    return [...scores.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK)
      .map(([index]) => index);
  }

  matchVerse(
    text: string,
    threshold = 0.3,
    maxSpan = 3,
    hint?: [number, number] | null,
    returnTopK = 0,
    useTrigramIndex = false,
  ): MatchVerseResult | null {
    const normalized = normalizeArabic(text);
    if (!normalized.trim()) return null;
    const bonuses = this.continuationBonuses(hint);

    const candidateIndexes = useTrigramIndex
      ? (() => {
          const indexes = new Set(this.trigramCandidates(normalized, 50));
          for (const ref of bonuses.keys()) {
            const idx = this.refToIndex.get(ref);
            if (idx !== undefined) indexes.add(idx);
          }
          return indexes.size < 20 ? this.verses.map((_, index) => index) : [...indexes];
        })()
      : this.verses.map((_, index) => index);

    const scored = candidateIndexes.map((index) => {
      const verse = this.verses[index];
      let raw = this.bestFragmentScore(normalized, verse);
      if (verse.text_clean_no_bsm) {
        let stripped = ratio(normalized, verse.text_clean_no_bsm);
        stripped = this.fragmentScore(normalized, verse.text_clean_no_bsm, stripped);
        raw = Math.max(raw, stripped);
      }
      const bonus = bonuses.get(`${verse.surah}:${verse.ayah}`) ?? 0;
      if (bonus > 0) {
        raw = Math.max(
          raw,
          this.suffixPrefixScore(normalized, verse.text_clean),
          this.suffixPrefixScore(normalized, verse.text_clean_alt),
        );
      }
      return { verse, raw, bonus, total: Math.min(raw + bonus, 1) };
    }).sort((a, b) => b.total - a.total);

    if (scored.length === 0) return null;

    let bestScore = scored[0].total;
    let best: MatchVerseResult = {
      surah: scored[0].verse.surah,
      ayah: scored[0].verse.ayah,
      ayah_end: scored[0].verse.ayah,
      text_clean: scored[0].verse.text_clean,
      score: bestScore,
      raw_score: scored[0].raw,
      bonus: scored[0].bonus,
    };

    const topSingles: RunnerUp[] = scored.slice(0, Math.max(returnTopK, 5)).map(({ verse, raw, bonus, total }) => ({
      surah: verse.surah,
      ayah: verse.ayah,
      raw_score: Number(raw.toFixed(3)),
      bonus: Number(bonus.toFixed(3)),
      score: Number(total.toFixed(3)),
      text_clean: verse.text_clean.slice(0, 60),
    }));

    const seenSurahs = new Set<number>();
    for (const { verse } of scored.slice(0, 20)) {
      if (seenSurahs.has(verse.surah)) continue;
      seenSurahs.add(verse.surah);
      const verses = this.getSurah(verse.surah) as QuranVerseInternal[];
      for (let i = 0; i < verses.length; i++) {
        for (let span = 2; span <= maxSpan; span++) {
          if (i + span > verses.length) break;
          const chunk = verses.slice(i, i + span);
          const firstText = chunk[0].text_clean_no_bsm || chunk[0].text_clean;
          const combined = [firstText, ...chunk.slice(1).map((item) => item.text_clean)].join(" ");
          const raw = ratio(normalized, combined);
          const bonus = bonuses.get(`${chunk[0].surah}:${chunk[0].ayah}`) ?? 0;
          const score = Math.min(raw + bonus, 1);
          if (score > bestScore) {
            bestScore = score;
            best = {
              surah: verse.surah,
              ayah: chunk[0].ayah,
              ayah_end: chunk[chunk.length - 1].ayah,
              text_clean: combined,
              score,
              raw_score: raw,
              bonus,
            };
          }
        }
      }
    }

    if (bestScore < threshold) return null;
    if (returnTopK > 0) best.runners_up = topSingles.slice(0, returnTopK);
    return best;
  }

  search(text: string, topK = 5): QuranMatch[] {
    const normalized = normalizeArabic(text);
    if (!normalized) return [];
    return this.verses
      .map((verse) => ({
        ...verse,
        score: Math.max(
          this.bestTextRatio(normalized, verse),
          ratio(normalized.replace(/ /g, ""), verse.text_clean.replace(/ /g, "")),
          ratio(normalized.replace(/ /g, ""), verse.text_clean_alt.replace(/ /g, "")),
        ),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  buildCandidates(text: string, topK = 100, maxSpan = 6, topSpanRefs = 80): { base: QuranCandidate | null; candidates: QuranCandidate[] } {
    const normalized = normalizeArabic(text);
    const seen = new Set<string>();
    const candidates: QuranCandidate[] = [];
    const singleRefs: Array<[number, number]> = [];

    const addCandidate = (candidate: QuranCandidate) => {
      const key = `${candidate.surah}:${candidate.ayah}:${candidate.ayah_end}`;
      if (seen.has(key)) return;
      seen.add(key);
      candidates.push(candidate);
      singleRefs.push([candidate.surah, candidate.ayah]);
    };

    const base = this.matchVerse(text, 0, maxSpan, undefined, topK, true);
    if (base) {
      addCandidate(base);
      for (const runnerUp of base.runners_up ?? []) {
        const verse = this.getVerse(runnerUp.surah, runnerUp.ayah);
        if (!verse) continue;
        addCandidate({
          surah: verse.surah,
          ayah: verse.ayah,
          ayah_end: verse.ayah,
          text_clean: verse.text_clean,
          score: runnerUp.score,
        });
      }
    }

    for (const verse of this.search(text, topK)) {
      addCandidate({
        surah: verse.surah,
        ayah: verse.ayah,
        ayah_end: verse.ayah,
        text_clean: verse.text_clean,
        score: verse.score,
      });
    }

    const spaceless = normalized.replace(/ /g, "");
    const fullScan = this.verses
      .map((verse) => ({
        verse,
        score: Math.max(
          this.bestTextRatio(normalized, verse),
          ratio(spaceless, verse.text_clean.replace(/ /g, "")),
          ratio(spaceless, verse.text_clean_alt.replace(/ /g, "")),
        ),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);

    for (const { verse, score } of fullScan) {
      addCandidate({
        surah: verse.surah,
        ayah: verse.ayah,
        ayah_end: verse.ayah,
        text_clean: verse.text_clean,
        score,
      });
    }

    for (const [surah, ayah] of singleRefs.slice(0, topSpanRefs)) {
      const verses = this.getSurah(surah);
      const maxAyah = verses.length;
      for (let start = Math.max(1, ayah - maxSpan + 1); start <= Math.min(ayah, maxAyah); start++) {
        for (let end = Math.max(ayah, start + 1); end <= Math.min(maxAyah, start + maxSpan - 1); end++) {
          const chunk = verses.slice(start - 1, end);
          if (chunk.length !== end - start + 1) continue;
          const first = (chunk[0] as QuranVerseInternal).text_clean_no_bsm || chunk[0].text_clean;
          const textClean = [first, ...chunk.slice(1).map((verse) => verse.text_clean)].join(" ");
          addCandidate({
            surah,
            ayah: start,
            ayah_end: end,
            text_clean: textClean,
            score: ratio(normalized, textClean),
          });
        }
      }
    }

    return {
      base: base,
      candidates,
    };
  }
}
