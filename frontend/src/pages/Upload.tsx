import { useEffect, useState } from "react";
import { api, type Job } from "../lib/api";
import { useApp } from "../store/app";
import { useExercises } from "../useExercises";

export function Upload() {
  const userId = useApp((s) => s.profile.userId);
  const sel = useApp((s) => s.selection);
  const { exercises } = useExercises();
  const [file, setFile] = useState<File | null>(null);
  const [exerciseId, setExerciseId] = useState(sel.exerciseId);
  const [side, setSide] = useState<"L" | "R">(sel.side);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;
    const t = setInterval(() => {
      api
        .job(jobId)
        .then((j) => {
          setJob(j);
          if (j.status === "done" || j.status === "error") clearInterval(t);
        })
        .catch((e: unknown) => setError(String(e)));
    }, 1000);
    return () => clearInterval(t);
  }, [jobId]);

  const submit = async () => {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    form.append("user_id", userId);
    form.append("exercise_id", exerciseId);
    form.append("side", side);
    setError(null);
    setJob(null);
    try {
      setJobId((await api.uploadVideo(form)).job_id);
    } catch (e) {
      setError(String(e));
    }
  };

  return (
    <section>
      <h1>Analyze a video</h1>
      <p className="muted">The video is processed on this computer's RehabPose server and deleted after analysis; the annotated result is kept with the session.</p>
      <div className="card row">
        <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        <select value={exerciseId} onChange={(e) => setExerciseId(e.target.value)}>
          {exercises.map((e) => <option key={e.id} value={e.id}>{e.name}</option>)}
        </select>
        <select value={side} onChange={(e) => setSide(e.target.value as "L" | "R")}>
          <option value="R">Right arm</option>
          <option value="L">Left arm</option>
        </select>
        <button className="primary" disabled={!file} onClick={submit}>Upload</button>
      </div>
      {error && <p className="error">{error}</p>}
      {job && (
        <div className="card" style={{ marginTop: 12 }}>
          <p>Status: {job.status} {job.error && <span className="error">{job.error}</span>}</p>
          <div className="progressbar" role="progressbar" aria-valuenow={Math.round(job.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
            <div style={{ width: `${Math.round(job.progress * 100)}%` }} />
          </div>
          {job.status === "done" && (
            <>
              <video src={job.result.annotated_video} controls style={{ width: "100%", marginTop: 12 }} />
              <p className="row">
                <a href={`#/summary/${job.session_id}`}>Summary</a>
                <a href={job.result.report_pdf} target="_blank" rel="noreferrer">PDF report</a>
              </p>
            </>
          )}
        </div>
      )}
    </section>
  );
}
