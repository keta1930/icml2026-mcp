import Image from "next/image";
import Link from "next/link";
import { siteConfig } from "@/lib/site";

const features = [
  ["Hybrid retrieval", "Chroma dense 与 SQLite FTS5 BM25 融合排序。"],
  ["Persistent sessions", "跨多轮 query 自动隐藏已经返回的论文。"],
  ["Local-first", "语料、索引与 session state 全部本地持久化。"],
];

export default function HomePage() {
  return (
    <main className="relative flex-1 overflow-hidden">
      <div className="hero-grid pointer-events-none absolute inset-0" />
      <section className="relative mx-auto max-w-7xl px-6 pb-16 pt-20 lg:px-8 lg:pt-28">
        <div className="mx-auto max-w-4xl text-center">
          <p className="mb-5 text-sm font-semibold tracking-[0.18em] text-cyan-500 uppercase">
            6,341 ICML 2026 papers · MCP native
          </p>
          <h1 className="text-balance text-5xl font-semibold tracking-tight sm:text-7xl">
            让 agent 从论文海洋中
            <span className="gradient-text block">持续找到新答案</span>
          </h1>
          <p className="mx-auto mt-7 max-w-2xl text-pretty text-lg text-fd-muted-foreground sm:text-xl">
            本地混合检索、持久化调研会话和有界公共服务，专为多轮研究工作流设计。
          </p>
          <div className="mt-9 flex flex-wrap justify-center gap-3">
            <Link
              href="/docs/getting-started/public-service"
              className="rounded-full bg-fd-primary px-6 py-3 font-medium text-fd-primary-foreground transition hover:opacity-90"
            >
              连接公共服务
            </Link>
            <Link
              href="/docs/getting-started/local-deployment"
              className="rounded-full border border-fd-border bg-fd-card px-6 py-3 font-medium transition hover:bg-fd-accent"
            >
              本地部署
            </Link>
          </div>
          <code className="mt-5 inline-block max-w-full overflow-x-auto rounded-lg border border-fd-border bg-fd-card/80 px-4 py-2 text-sm text-fd-muted-foreground">
            {siteConfig.publicMcpUrl}
          </code>
        </div>

        <div className="relative mx-auto mt-16 max-w-6xl overflow-hidden rounded-3xl border border-white/10 bg-[#071629] shadow-2xl shadow-cyan-950/30">
          <Image
            src={`${siteConfig.basePath}/images/vintage-paper-search-cover.png`}
            alt="ICML Paper Search MCP 复古科学期刊风格封面"
            width={1672}
            height={941}
            priority
            className="h-auto w-full sepia-[0.08]"
          />
        </div>

        <div className="mx-auto mt-14 grid max-w-6xl gap-4 md:grid-cols-3">
          {features.map(([title, description]) => (
            <div key={title} className="rounded-2xl border border-fd-border bg-fd-card/70 p-6 backdrop-blur">
              <h2 className="font-semibold">{title}</h2>
              <p className="mt-2 text-sm leading-6 text-fd-muted-foreground">{description}</p>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
