import { ReviewPanel } from "@/components/ReviewPanel";

export default function ReviewPage() {
  return (
    <>
      <h1 style={{ fontSize: 22, margin: "0 0 4px" }}>Review teacher labels</h1>
      <p className="meta" style={{ marginTop: 0 }}>
        Accept or correct the labels the teacher model proposed for each filing. Reviewed labels become the gold set every model is evaluated on.
      </p>
      <ReviewPanel />
    </>
  );
}
