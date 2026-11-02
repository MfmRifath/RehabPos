import { CircleAlert, CircleCheck, Download, FileVideo, ListChecks, LoaderCircle, UploadCloud } from "lucide-react";
import { useEffect, useState, type DragEvent } from "react";
import { Segmented } from "../components/ui";
import { api, type Job } from "../lib/api";
import { describeApiError } from "../lib/errors";
import { formatBytes } from "../lib/format";
import { useApp, type Side } from "../store/app";
import { useExercises } from "../useExercises";

const JOB_TEXT: Record<Job["status"], string> = {
  queued: "Waiting to start…",
  running: "Analyzing your video…",
  done: "Analysis finished",
  error: "The analysis didn't finish",
};

const SIDES = [
  { value: "R" as Side, label: "Right arm" },
  { value: "L" as Side, label: "Left arm" },
];

export function Upload() {
  const userId = useApp((s) => s.profile.userId);
  const sel = useApp((s) => s.selection);
  const { exercises } = useExercises();
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [exerciseId, setExerciseId] = useState(sel.exerciseId);
  const [side, setSide] = useState<Side>(sel.side);
  const [uploading, setUploading] = useState(false);
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
        .catch((e: unknown) => {
          setError(describeApiError(e));
          clearInterval(t);
        });
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
    setJobId(null);
    setUploading(true);
    try {
      setJobId((await api.uploadVideo(form)).job_id);
    } catch (e) {
      setError(describeApiError(e));
    } finally {
      setUploading(false);
    }
  };

  const onDrop = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (!dropped) return;
    if (dropped.type.startsWith("video/")) {
      setFile(dropped);
      setError(null);
    } else setError(`"${dropped.name}" isn't a video file.`);
  };

  const working = uploading || (jobId !== null && job?.status !== "done" && job?.status !== "error" && !error);

  return (
    <section>
      <div className="page-head">
        <h1>Analyze a video</h1>
        <p className="lede">The video is processed on this computer's RehabPose server and deleted after analysis; the annotated result is kept with the session.</p>
      </div>
      <div className="card upload-card">
        <label
          className={`dropzone ${dragging ? "dragging" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <input className="sr-only" type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} data-testid="video-file" />
          {file ? <FileVideo size={32} aria-hidden="true" /> : <UploadCloud size={32} aria-hidden="true" />}
          {file ? <strong>{file.name}</strong> : <strong>Drop a video here, or click to choose one</strong>}
          <span className="small">{file ? `${formatBytes(file.size)} · click to choose a different file` : "Any video your browser can read"}</span>
        </label>
        <div className="field-row">
          <label className="field">
            Exercise
            <select className="select" value={exerciseId} onChange={(e) => setExerciseId(e.target.value)}>
              {exercises.map((e) => <option key={e.id} value={e.id}>{e.name}</option>)}
            </select>
          </label>
          <Segmented legend="Which arm?" name="upload-side" value={side} options={SIDES} onChange={setSide} />
        </div>
        <button className="btn btn-primary btn-lg" disabled={!file || working} onClick={submit}>
          {uploading ? <LoaderCircle size={20} className="spin" aria-hidden="true" /> : <UploadCloud size={20} aria-hidden="true" />}
          {uploading ? "Uploading…" : "Upload and analyze"}
        </button>
      </div>
      {error && (
        <p className="note note-critical job-card" role="alert">
          <CircleAlert size={20} aria-hidden="true" />
          <span>{error}</span>
        </p>
      )}
      {job && (
        <div className="card job-card" role="status">
          <p className="job-status">
            {job.status === "done" ? <CircleCheck size={20} aria-hidden="true" /> : job.status === "error" ? <CircleAlert size={20} aria-hidden="true" /> : <LoaderCircle size={20} className="spin" aria-hidden="true" />}
            {JOB_TEXT[job.status]}
          </p>
          {job.error && <p className="field-error">{job.error}</p>}
          {job.status !== "done" && job.status !== "error" && (
            <div className="progressbar" role="progressbar" aria-label="Analysis progress" aria-valuenow={Math.round(job.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
              <div style={{ width: `${Math.round(job.progress * 100)}%` }} />
            </div>
          )}
          {job.status === "done" && (
            <>
              <video className="job-video" src={job.result.annotated_video} controls />
              <div className="btn-row">
                <a className="btn btn-primary" href={`#/summary/${job.session_id}`}><ListChecks size={18} aria-hidden="true" />View summary</a>
                <a className="btn" href={job.result.report_pdf} target="_blank" rel="noreferrer"><Download size={18} aria-hidden="true" />PDF report</a>
              </div>
            </>
          )}
        </div>
      )}
    </section>
  );
}
