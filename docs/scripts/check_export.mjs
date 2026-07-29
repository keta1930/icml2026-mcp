import { access, readFile, readdir } from "node:fs/promises";
import path from "node:path";

const languages = ["en", "zh"];
const contentRoot = path.resolve("content/docs");
const outputRoot = path.resolve("out");
const expectedPageCount = 21;

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function exists(target) {
  try {
    await access(target);
    return true;
  } catch {
    return false;
  }
}

async function walk(directory, suffix) {
  const results = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) results.push(...(await walk(target, suffix)));
    else if (entry.name.endsWith(suffix)) results.push(target);
  }
  return results;
}

function relativePages(language) {
  return walk(path.join(contentRoot, language), ".mdx").then((files) =>
    files
      .map((file) => path.relative(path.join(contentRoot, language), file))
      .sort(),
  );
}

function pageSlug(relative) {
  const normalized = relative.split(path.sep).join("/").replace(/\.mdx$/, "");
  return normalized === "index" ? "" : normalized;
}

function docsRoute(language, slug) {
  return `/${language}/docs${slug ? `/${slug}` : ""}`;
}

async function requireFile(target, label) {
  assert(await exists(target), `${label} is missing: ${path.relative(outputRoot, target)}`);
}

const pagesByLanguage = Object.fromEntries(
  await Promise.all(
    languages.map(async (language) => [language, await relativePages(language)]),
  ),
);
for (const language of languages) {
  assert(
    pagesByLanguage[language].length === expectedPageCount,
    `${language} source page count is ${pagesByLanguage[language].length}, expected ${expectedPageCount}`,
  );
}
assert(
  JSON.stringify(pagesByLanguage.en) === JSON.stringify(pagesByLanguage.zh),
  "source locale page parity failed during export validation",
);

const githubPages = process.env.GITHUB_PAGES === "true";
const configuredBasePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "/icml2026-mcp";
const basePath = githubPages ? configuredBasePath.replace(/\/$/, "") : "";
const siteRoot = (
  process.env.NEXT_PUBLIC_SITE_URL ?? "http://127.0.0.1:20440"
).replace(/\/$/, "");
if (basePath) {
  assert(siteRoot.endsWith(basePath), "NEXT_PUBLIC_SITE_URL must end with the Pages base path");
}

await requireFile(path.join(outputRoot, "index.html"), "language entry page");
const rootLlmsPath = path.join(outputRoot, "llms.txt");
await requireFile(rootLlmsPath, "root AI index");
await requireFile(path.join(outputRoot, "api/search"), "static search database");
assert(!(await exists(path.join(outputRoot, "docs"))), "root /docs output must not exist");

for (const language of languages) {
  await requireFile(path.join(outputRoot, language, "llms.txt"), `${language} AI index`);
  assert(
    !(await exists(path.join(outputRoot, language, "llms.mdx"))),
    `${language} internal llms.mdx route leaked into the published output`,
  );
  for (const relative of pagesByLanguage[language]) {
    const slug = pageSlug(relative);
    const htmlPath = slug
      ? path.join(outputRoot, language, "docs", slug, "index.html")
      : path.join(outputRoot, language, "docs", "index.html");
    const markdown = slug
      ? path.join(outputRoot, language, "docs", `${slug}.md`)
      : path.join(outputRoot, language, "docs.md");
    await requireFile(htmlPath, `${language} HTML page`);
    await requireFile(markdown, `${language} Markdown page`);

    const html = await readFile(htmlPath, "utf8");
    const htmlLanguage = language === "zh" ? "zh-CN" : "en";
    const route = docsRoute(language, slug);
    const canonical = `${siteRoot}${route}/`;
    assert(
      html.includes(`<html lang="${htmlLanguage}">`),
      `${language} html lang is incorrect for ${route}`,
    );
    assert(
      html.includes("English") && html.includes("简体中文"),
      `${route} does not expose both language names`,
    );
    assert(
      html.includes(`<link rel="canonical" href="${canonical}"/>`),
      `canonical URL is incorrect for ${route}`,
    );
    for (const [alternateLanguage, routeLanguage] of [
      ["en", "en"],
      ["zh-CN", "zh"],
    ]) {
      const alternate = `${siteRoot}${docsRoute(routeLanguage, slug)}/`;
      assert(
        html.includes(
          `<link rel="alternate" hrefLang="${alternateLanguage}" href="${alternate}"/>`,
        ),
        `${route} lacks ${alternateLanguage} alternate metadata`,
      );
    }
    assert(
      html.includes(`${basePath}/_next/`),
      `${route} assets do not use the expected base path`,
    );
    if (basePath) {
      assert(
        !html.includes(`${basePath}${basePath}/`),
        `${route} contains a duplicated base path`,
      );
      assert(
        html.includes(`href="${basePath}${route}/"`),
        `${route} does not expose its Pages-prefixed navigation path`,
      );
    }
  }
}

const search = JSON.parse(await readFile(path.join(outputRoot, "api/search"), "utf8"));
assert(search.type === "i18n", "search export must use the i18n database format");
assert(
  JSON.stringify(Object.keys(search.data ?? {}).sort()) === JSON.stringify(languages),
  "search export must contain exactly the en and zh databases",
);
for (const language of languages) {
  const serialized = JSON.stringify(search.data[language]);
  const otherLanguage = language === "en" ? "zh" : "en";
  for (const relative of pagesByLanguage[language]) {
    const slug = pageSlug(relative);
    const route = docsRoute(language, slug);
    assert(serialized.includes(route), `${language} search database lacks ${route}`);
  }
  assert(
    !serialized.includes(`/${otherLanguage}/docs`),
    `${language} search database contains ${otherLanguage} routes`,
  );
}

const rootLlms = await readFile(rootLlmsPath, "utf8");
for (const language of languages) {
  assert(
    rootLlms.includes(`${siteRoot}/${language}/llms.txt`),
    `root AI index lacks the ${language} locale index`,
  );
}

for (const language of languages) {
  const localeLlms = await readFile(
    path.join(outputRoot, language, "llms.txt"),
    "utf8",
  );
  const otherLanguage = language === "en" ? "zh" : "en";
  const llmsEntries = localeLlms.match(/^- \[/gm) ?? [];
  assert(
    llmsEntries.length === expectedPageCount,
    `${language} AI index has ${llmsEntries.length} pages, expected ${expectedPageCount}`,
  );
  assert(
    !localeLlms.includes(`${siteRoot}/${otherLanguage}/docs`),
    `${language} AI index contains ${otherLanguage} pages`,
  );
  for (const relative of pagesByLanguage[language]) {
    const slug = pageSlug(relative);
    const markdownUrl = slug
      ? `${siteRoot}/${language}/docs/${slug}.md`
      : `${siteRoot}/${language}/docs.md`;
    assert(
      localeLlms.includes(markdownUrl),
      `${language} AI index lacks ${markdownUrl}`,
    );

    const markdownPath = slug
      ? path.join(outputRoot, language, "docs", `${slug}.md`)
      : path.join(outputRoot, language, "docs.md");
    const markdown = await readFile(markdownPath, "utf8");
    assert(
      !markdown.includes(`](/${language}/docs`) && !markdown.includes("](/docs"),
      `${language} Markdown contains a deployment-relative documentation link`,
    );
    assert(
      !markdown.includes(`${siteRoot}/${otherLanguage}/docs`),
      `${language} Markdown contains a ${otherLanguage} documentation link`,
    );
  }
}

process.stdout.write(
  `checked export: ${expectedPageCount} HTML and Markdown pages per locale, bilingual search and metadata\n`,
);
