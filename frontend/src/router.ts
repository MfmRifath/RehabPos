import { useEffect, useState } from "react";

/** Current hash route split into segments, e.g. "#/summary/abc" -> ["summary", "abc"]. */
export function useRoute(): string[] {
  const read = () => window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export const navigate = (path: string): void => {
  window.location.hash = path.startsWith("/") ? path : `/${path}`;
};
