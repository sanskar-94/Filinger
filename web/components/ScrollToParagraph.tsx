"use client";

import { useEffect } from "react";

export function ScrollToParagraph({ id }: { id: string }) {
  useEffect(() => {
    document.getElementById(id)?.scrollIntoView({ block: "center" });
  }, [id]);
  return null;
}
