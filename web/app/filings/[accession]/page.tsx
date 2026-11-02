import Link from "next/link";
import { ScrollToParagraph } from "@/components/ScrollToParagraph";
import { backendJson } from "@/lib/backend";
import { sectionLabel } from "@/lib/citations";
import type { FilingDocument } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function FilingPage({
  params,
  searchParams,
}: {
  params: Promise<{ accession: string }>;
  searchParams: Promise<{ p?: string }>;
}) {
  const { accession } = await params;
  const { p } = await searchParams;
  const doc = await backendJson<FilingDocument>(`/v1/filings/${encodeURIComponent(accession)}/paragraphs`);
  if (!doc) return <p className="error">Filing {accession} was not found.</p>;
  return (
    <>
      <p className="meta">
        <Link href={doc.ticker ? `/companies/${doc.ticker}` : "/companies"}>{doc.company} ({doc.ticker})</Link> · {doc.form} filed{" "}
        {doc.date_filed} · <code>{doc.accession}</code> ·{" "}
        <a href={doc.url} target="_blank" rel="noreferrer">
          Original on EDGAR
        </a>
      </p>
      <h1 style={{ fontSize: 22, marginTop: 4 }}>
        {doc.company} {doc.form}
      </h1>
      <nav className="chips" aria-label="Sections">
        {doc.sections.map((s) => (
          <a key={s.section} className="chip" href={`#s-${s.section}`}>
            {sectionLabel(s.section)}
          </a>
        ))}
      </nav>
      {doc.sections.map((s) => (
        <section key={s.section} id={`s-${s.section}`} className="card" style={{ marginTop: 16 }}>
          <h2>{sectionLabel(s.section)}</h2>
          {s.paragraphs.map((para) => (
            <p key={para.paragraph_id} id={`p-${para.paragraph_id}`} className={`para${para.paragraph_id === p ? " cited" : ""}`}>
              <span className="pid">¶{para.para_idx + 1}</span>
              {para.text}
            </p>
          ))}
        </section>
      ))}
      {p && <ScrollToParagraph id={`p-${p}`} />}
    </>
  );
}
