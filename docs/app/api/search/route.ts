import { createFromSource } from "fumadocs-core/search/server";
import { createStableMandarinTokenizer } from "@/lib/search-tokenizer";
import { source } from "@/lib/source";

export const revalidate = false;

export const { staticGET: GET } = createFromSource(source, {
  tokenizer: createStableMandarinTokenizer(),
  search: {
    threshold: 0,
    tolerance: 0,
  },
});
