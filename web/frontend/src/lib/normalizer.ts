const DIACRITICS = /[\u064B-\u065F]/g;
const QURANIC_ANNOTATIONS = /[\u06D6-\u06ED]/g;
const ORNATE_PARENS = /[\uFD3E\uFD3F]/g;
const ARABIC_DIGITS = /[\u0660-\u0669\u06F0-\u06F9]/g;
const PUNCTUATION = /[.,;:!?\u2026\u060C\u061B\u061F]/g;

export function normalizeArabic(text: string): string {
  return String(text)
    .replace(/\uFEFF|\u200F|\u200E/g, "")
    .replace(DIACRITICS, "")
    .replace(/\u0622|\u0671|[\u0672\u0673]|\u0670/g, "ا")
    .replace(/[\u06CC\u06D2]/g, "ي")
    .replace(/\u06A9/g, "ك")
    .replace(QURANIC_ANNOTATIONS, "")
    .replace(ORNATE_PARENS, "")
    .replace(ARABIC_DIGITS, "")
    .replace(/\u0640/g, "")
    .replace(PUNCTUATION, "")
    .replace(/\s+/g, " ")
    .trim();
}
