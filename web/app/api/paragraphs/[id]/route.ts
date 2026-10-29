import { NextResponse } from "next/server";
import { backend } from "@/lib/backend";

export async function GET(_req: Request, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  if (!/^[0-9a-f]{16}$/.test(id)) return NextResponse.json({ error: "bad id" }, { status: 400 });
  const r = await backend(`/v1/paragraphs/${id}`);
  return NextResponse.json(await r.json(), { status: r.status });
}
