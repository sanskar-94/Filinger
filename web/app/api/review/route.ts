import { NextResponse, type NextRequest } from "next/server";
import { backend } from "@/lib/backend";

export async function POST(req: NextRequest) {
  const body = (await req.json()) as Record<string, unknown>;
  const reviewer = req.cookies.get("flab_uid")?.value ?? "anonymous";
  const r = await backend("/v1/review", { method: "POST", body: JSON.stringify({ ...body, reviewer }) });
  return NextResponse.json(await r.json(), { status: r.status });
}
