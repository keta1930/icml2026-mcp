import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared";
import { siteConfig } from "@/lib/site";

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: siteConfig.name,
    },
    links: [
      {
        text: "文档",
        url: "/docs",
        active: "nested-url",
      },
      {
        text: "公共 MCP",
        url: siteConfig.publicMcpUrl,
        external: true,
      },
    ],
  };
}
