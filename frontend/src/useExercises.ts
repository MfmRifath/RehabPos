import { useCallback, useEffect, useState } from "react";
import { api, type Exercise } from "./lib/api";
import { describeApiError } from "./lib/errors";

let cache: Exercise[] | null = null;

export function useExercises(): { exercises: Exercise[]; error: string | null; loading: boolean; reload: () => void } {
  const [exercises, setExercises] = useState<Exercise[]>(cache ?? []);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(cache === null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (cache) return;
    api
      .exercises()
      .then((e) => {
        cache = e;
        setExercises(e);
        setLoading(false);
      })
      .catch((err: unknown) => {
        setError(describeApiError(err));
        setLoading(false);
      });
  }, [attempt]);
  const reload = useCallback(() => {
    setError(null);
    setLoading(true);
    setAttempt((a) => a + 1);
  }, []);
  return { exercises, error, loading, reload };
}
