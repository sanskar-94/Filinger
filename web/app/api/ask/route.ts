import { NextResponse, type NextRequest } from "next/server";
import { backend } from "@/lib/backend";
import { userIdFrom } from "@/lib/uid";

export async function POST(req: NextRequest) {
  const body = (await req.json()) as { question?: string; ticker?: string | null };
  if (!body.question || body.question.trim().length < 3) {
    return NextResponse.json({ error: "question is required" }, { status: 422 });
  }
  const user_id = userIdFrom(req.headers);
  const r = await backend("/v1/agent/ask", {
    method: "POST",
    body: JSON.stringify({ question: body.question, ticker: body.ticker || null, user_id }),
  });
  return NextResponse.json(await r.json(), { status: r.status });
}
