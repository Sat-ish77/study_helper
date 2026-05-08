"use client";

import { useEffect } from "react";
import { useAppStore } from "@/store/useAppStore";

export function ThemeInitializer() {
  const theme = useAppStore((s) => s.theme);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  return null;
}
