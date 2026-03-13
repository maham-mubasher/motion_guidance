"use client";

import { useEffect, useMemo, useState } from "react";

type JobResponse = {
  status?: string;
  progress?: number;
  message?: string;
  code?: number;
};

export default function Home() {
  const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

  const [prompt, setPrompt] = useState("a teapot floating in water");
  const [inputDir, setInputDir] = useState("./data/teapot");
  const [mask, setMask] = useState("down150.mask.pth");
  const [flow, setFlow] = useState("down150.pth");
  const [fastPreview, setFastPreview] = useState(true);
  const [ddimSteps, setDdimSteps] = useState(20);
  const [guidanceWeight, setGuidanceWeight] = useState(30);
  const [clipGrad, setClipGrad] = useState(60);
  const [raftIters, setRaftIters] = useState(1);
  const [recursiveSteps, setRecursiveSteps] = useState(1);

  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("idle");
  const [message, setMessage] = useState<string>("ready");
  const [progress, setProgress] = useState<number>(0);
  const [resultsPrefix, setResultsPrefix] = useState<string>("");

  const isRunning = status === "queued" || status === "running";
  const progressSafe = Math.max(0, Math.min(100, progress));

  const previewUrl = useMemo(() => {
    if (!resultsPrefix) return "";
    return `${resultsPrefix}sample_000/pred.png`;
  }, [resultsPrefix]);

  useEffect(() => {
    if (!jobId || !isRunning) return;

    const intervalId = window.setInterval(async () => {
      try {
        const res = await fetch(`${API}/jobs/${jobId}`);
        if (!res.ok) return;

        const data = (await res.json()) as JobResponse;
        if (typeof data.status === "string") setStatus(data.status);
        if (typeof data.message === "string") setMessage(data.message);
        if (typeof data.progress === "number") setProgress(data.progress);

        if (data.status === "done") {
          setMessage("completed");
          setProgress(100);
          window.clearInterval(intervalId);
        }
        if (data.status === "failed") {
          setMessage(data.code ? `failed (code ${data.code})` : "failed");
          window.clearInterval(intervalId);
        }
        if (data.status === "not_found") {
          setStatus("failed");
          setProgress(0);
          setMessage(data.message || "job state lost (backend restarted), please rerun");
          window.clearInterval(intervalId);
        }
      } catch {
        setMessage("lost connection to backend");
      }
    }, 1200);

    return () => window.clearInterval(intervalId);
  }, [API, jobId, isRunning]);

  const runJob = async () => {
    setStatus("queued");
    setMessage("submitting");
    setProgress(0);
    setJobId(null);
    setResultsPrefix("");

    try {
      const res = await fetch(`${API}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          input_dir: inputDir,
          edit_mask_path: mask,
          target_flow_name: flow,
          use_cached_latents: true,
          log_freq: fastPreview ? 0 : 5,
          ddim_steps: fastPreview ? Math.max(ddimSteps, 20) : Math.max(ddimSteps, 25),
          guidance_weight: guidanceWeight,
          clip_grad: clipGrad,
          raft_iters: raftIters,
          scale: 6.5,
          num_recursive_steps: recursiveSteps,
        }),
      });

      if (!res.ok) {
        setStatus("failed");
        setMessage(`failed (${res.status})`);
        return;
      }

      const data = await res.json();
      setJobId(data.job_id || null);
      setStatus(data.status || "running");
      setMessage("starting");
      setProgress(typeof data.progress === "number" ? data.progress : 0);
      setResultsPrefix(`${API}${data.results_url_prefix || ""}`);
    } catch {
      setStatus("failed");
      setMessage("backend unreachable (start API on http://localhost:8000)");
    }
  };

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_20%_15%,#1f8f88_0%,transparent_35%),radial-gradient(circle_at_85%_0%,#d38b30_0%,transparent_32%),linear-gradient(160deg,#0a0f15_0%,#101925_60%,#0d141e_100%)] px-4 py-8 text-slate-100 md:px-8">
      <div className="mx-auto max-w-5xl rounded-3xl border border-white/15 bg-white/8 p-6 shadow-[0_20px_80px_rgba(0,0,0,0.45)] backdrop-blur md:p-8">
        <div className="mb-6 flex items-center justify-between gap-3">
          <h1 className="text-2xl font-semibold tracking-tight md:text-3xl">Motion Guidance Console</h1>
          <div className="rounded-full border border-emerald-300/30 bg-emerald-300/10 px-3 py-1 text-xs font-semibold text-emerald-200">
            {status.toUpperCase()}
          </div>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          <section className="space-y-3 rounded-2xl border border-white/10 bg-slate-950/45 p-4">
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">Prompt</div>
              <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
            </label>
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">Input Dir</div>
              <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" value={inputDir} onChange={(e) => setInputDir(e.target.value)} />
            </label>
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">Mask File</div>
              <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" value={mask} onChange={(e) => setMask(e.target.value)} />
            </label>
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">Target Flow</div>
              <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" value={flow} onChange={(e) => setFlow(e.target.value)} />
            </label>
          </section>

          <section className="space-y-3 rounded-2xl border border-white/10 bg-slate-950/45 p-4">
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">Mode</div>
              <select className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" value={fastPreview ? "fast" : "quality"} onChange={(e) => setFastPreview(e.target.value === "fast")}>
                <option value="fast">Fast Preview</option>
                <option value="quality">Better Quality</option>
              </select>
            </label>
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">DDIM Steps</div>
              <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" min={10} max={250} value={ddimSteps} onChange={(e) => setDdimSteps(Number(e.target.value))} />
            </label>
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">Guidance Weight</div>
              <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" min={1} max={400} value={guidanceWeight} onChange={(e) => setGuidanceWeight(Number(e.target.value))} />
            </label>
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">Clip Grad</div>
              <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" min={0} max={400} value={clipGrad} onChange={(e) => setClipGrad(Number(e.target.value))} />
            </label>
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">RAFT Iters</div>
              <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" min={1} max={5} value={raftIters} onChange={(e) => setRaftIters(Number(e.target.value))} />
            </label>
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">Recursive Steps</div>
              <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" min={1} max={2} value={recursiveSteps} onChange={(e) => setRecursiveSteps(Number(e.target.value))} />
            </label>

            <button
              onClick={runJob}
              disabled={isRunning}
              className="mt-2 w-full rounded-lg bg-[linear-gradient(120deg,#0ea5a4_0%,#14b8a6_45%,#f59e0b_100%)] px-4 py-2 font-semibold text-slate-950 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isRunning ? "Running..." : "Run Generation"}
            </button>
          </section>
        </div>

        <section className="mt-6 rounded-2xl border border-white/10 bg-slate-950/45 p-4">
          <div className="mb-2 flex items-center justify-between text-sm">
            <span className="text-slate-200">{message}</span>
            <span className="font-semibold">{progressSafe}%</span>
          </div>
          <div className="h-3 overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full rounded-full bg-[linear-gradient(90deg,#22d3ee_0%,#34d399_50%,#f59e0b_100%)] transition-all duration-500"
              style={{ width: `${progressSafe}%` }}
            />
          </div>
          <div className="mt-3 text-xs text-slate-300">Job ID: {jobId || "-"}</div>
          {previewUrl && (
            <div className="mt-3">
              <a className="text-sm font-medium text-cyan-300 underline" href={previewUrl} target="_blank" rel="noreferrer">
                Open latest output image
              </a>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
