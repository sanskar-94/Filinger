import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Filings Intelligence Lab",
  description: "Ask questions about any company's SEC filings, with citations to the exact paragraph.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="top">
          <span className="brand">Filings Intelligence Lab</span>
          <nav>
            <Link href="/">Ask</Link>
            <Link href="/companies">Companies</Link>
          </nav>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
