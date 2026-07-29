import { createTokenizer } from "@orama/tokenizers/mandarin";

const WORDS = /[\p{Script=Han}]|[\p{L}\p{N}_-]+/gu;

export function createStableMandarinTokenizer() {
  const tokenizer = createTokenizer();

  tokenizer.tokenize = (input, _language, prop, withCache = true) => {
    const normalize = tokenizer.normalizeToken.bind(tokenizer, prop ?? "");
    const rawTokens =
      prop && tokenizer.tokenizeSkipProperties.has(prop)
        ? [input]
        : (input.toLowerCase().match(WORDS) ?? []);
    const tokens = rawTokens
      .map((token) => normalize(token, withCache))
      .filter(Boolean);

    return tokenizer.allowDuplicates ? tokens : Array.from(new Set(tokens));
  };

  return tokenizer;
}
