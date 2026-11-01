import { useEffect, useState } from "react";
import { api, type Exercise } from "./lib/api";

let cache: Exercise[] | null = null;

export function useExercises(): { exercises: Exercise[]; error: string | null } {
  const [exercises, setExercises] = useState<Exercise[]>(cache ?? []);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (cache) return;
    api
      .exercises()
      .then((e) => {
        cache = e;
        setExercises(e);
      })
      .catch((err: unknown) => setError(`Cannot reach the RehabPose server: ${String(err)}`));
  }, []);
  return { exercises, error };
}
