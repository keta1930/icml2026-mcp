export function markdownUrl(pageUrl: string): string {
  return pageUrl === "/docs" ? "/docs.md" : `${pageUrl}.md`;
}
