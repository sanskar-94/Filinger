import { AskPanel } from "@/components/AskPanel";

export default function Home() {
  return (
    <>
      <h1 style={{ fontSize: 22, margin: "0 0 4px" }}>Ask the filings</h1>
      <p className="meta" style={{ marginTop: 0 }}>
        Answers cite the exact filing paragraph. Market reactions come from daily prices; model outputs are labelled as such.
      </p>
      <AskPanel />
    </>
  );
}
