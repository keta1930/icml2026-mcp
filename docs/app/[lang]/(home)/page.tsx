import Image from "next/image";
import Link from "next/link";
import { languages, requireLanguage } from "@/lib/i18n";
import { siteConfig, siteCopy } from "@/lib/site";

type PageProps = {
  params: Promise<{ lang: string }>;
};

export function generateStaticParams() {
  return languages.map((lang) => ({ lang }));
}

export default async function HomePage({ params }: PageProps) {
  const language = requireLanguage((await params).lang);
  const copy = siteCopy(language);
  return (
    <main className="relative flex-1 overflow-hidden">
      <div className="hero-grid pointer-events-none absolute inset-0" />
      <section className="relative mx-auto max-w-7xl px-6 pb-16 pt-20 lg:px-8 lg:pt-28">
        <div className="mx-auto max-w-4xl text-center">
          <p className="mb-5 text-sm font-semibold tracking-[0.18em] text-cyan-500 uppercase">
            {copy.homeEyebrow}
          </p>
          <h1 className="text-balance text-5xl font-semibold tracking-tight sm:text-7xl">
            {copy.homeTitlePrefix}
            <span className="gradient-text block">{copy.homeTitle}</span>
          </h1>
          <p className="mx-auto mt-7 max-w-2xl text-pretty text-lg text-fd-muted-foreground sm:text-xl">
            {copy.homeDescription}
          </p>
          <div className="mt-9 flex flex-wrap justify-center gap-3">
            <Link
              href={`/${language}/docs/getting-started/public-service`}
              className="rounded-full bg-fd-primary px-6 py-3 font-medium text-fd-primary-foreground transition hover:opacity-90"
            >
              {copy.publicService}
            </Link>
            <Link
              href={`/${language}/docs/getting-started/local-deployment`}
              className="rounded-full border border-fd-border bg-fd-card px-6 py-3 font-medium transition hover:bg-fd-accent"
            >
              {copy.localDeployment}
            </Link>
          </div>
          <code className="mt-5 inline-block max-w-full overflow-x-auto rounded-lg border border-fd-border bg-fd-card/80 px-4 py-2 text-sm text-fd-muted-foreground">
            {siteConfig.publicMcpUrl}
          </code>
        </div>

        <div className="relative mx-auto mt-16 max-w-6xl overflow-hidden rounded-3xl border border-white/10 bg-[#071629] shadow-2xl shadow-cyan-950/30">
          <Image
            src={`${siteConfig.basePath}/images/vintage-paper-search-cover.webp`}
            alt={copy.imageAlt}
            width={1672}
            height={941}
            priority
            className="h-auto w-full sepia-[0.08]"
          />
        </div>

        <div className="mx-auto mt-14 grid max-w-6xl gap-4 md:grid-cols-3">
          {copy.features.map(([title, description]) => (
            <div
              key={title}
              className="rounded-2xl border border-fd-border bg-fd-card/70 p-6 backdrop-blur"
            >
              <h2 className="font-semibold">{title}</h2>
              <p className="mt-2 text-sm leading-6 text-fd-muted-foreground">
                {description}
              </p>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
