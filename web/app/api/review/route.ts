import { NextResponse, type NextRequest } from "next/server";
import { backend } from "@/lib/backend";
import { userIdFrom } from "@/lib/uid";

export async function POST(req: NextRequest) {
  const body = (await req.json()) as Record<string, unknown>;
  const reviewer = userIdFrom(req.headers);
  const r = await backend("/v1/review", { method: "POST", body: JSON.stringify({ ...body, reviewer }) });
  return NextResponse.json(await r.json(), { status: r.status });
}
