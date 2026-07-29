import Link from "next/link";
import { siteConfig } from "@/lib/site";

export default function LanguageEntryPage() {
  return (
    <main className="hero-grid flex min-h-screen items-center justify-center px-6 py-16">
      <section className="w-full max-w-2xl rounded-3xl border border-fd-border bg-fd-card/90 p-10 text-center shadow-2xl">
        <p className="text-sm font-semibold tracking-[0.18em] text-cyan-500 uppercase">
          Documentation
        </p>
        <h1 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
          {siteConfig.name}
        </h1>
        <p className="mt-5 text-fd-muted-foreground">
          Choose a documentation language · 选择文档语言
        </p>
        <div className="mt-8 flex flex-wrap justify-center gap-3">
          <Link
            href="/en/docs"
            className="rounded-full bg-fd-primary px-6 py-3 font-medium text-fd-primary-foreground"
          >
            English
          </Link>
          <Link
            href="/zh/docs"
            className="rounded-full border border-fd-border px-6 py-3 font-medium hover:bg-fd-accent"
          >
            简体中文
          </Link>
        </div>
      </section>
    </main>
  );
}
