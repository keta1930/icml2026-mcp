"use client";

import { create } from "@orama/orama";
import { useMemo } from "react";
import { useDocsSearch } from "fumadocs-core/search/client";
import { oramaStaticClient } from "fumadocs-core/search/client/orama-static";
import {
  SearchDialog,
  SearchDialogClose,
  SearchDialogContent,
  SearchDialogHeader,
  SearchDialogIcon,
  SearchDialogInput,
  SearchDialogList,
  SearchDialogOverlay,
  type SharedProps,
} from "fumadocs-ui/components/dialog/search";
import { useI18n } from "fumadocs-ui/contexts/i18n";
import { createStableMandarinTokenizer } from "@/lib/search-tokenizer";
import { siteConfig } from "@/lib/site";

export default function StaticSearchDialog(props: SharedProps) {
  const { locale = "en" } = useI18n();
  const client = useMemo(
    () =>
      oramaStaticClient({
        from: `${siteConfig.basePath}/api/search`,
        locale,
        initOrama: (databaseLocale) =>
          databaseLocale === "zh"
            ? create({
                schema: { _: "string" },
                components: { tokenizer: createStableMandarinTokenizer() },
              })
            : create({ schema: { _: "string" }, language: "english" }),
        search: {
          threshold: 0,
          tolerance: 0,
        },
      }),
    [locale],
  );
  const { search, setSearch, query } = useDocsSearch({ client });

  return (
    <SearchDialog
      search={search}
      onSearchChange={setSearch}
      isLoading={query.isLoading}
      {...props}
    >
      <SearchDialogOverlay />
      <SearchDialogContent>
        <SearchDialogHeader>
          <SearchDialogIcon />
          <SearchDialogInput />
          <SearchDialogClose />
        </SearchDialogHeader>
        <SearchDialogList items={query.data !== "empty" ? query.data : null} />
      </SearchDialogContent>
    </SearchDialog>
  );
}
