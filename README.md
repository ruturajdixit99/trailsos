# Trail SOS

**A hiker with no cell signal sends a 5-second voice note and a photo across a LoRa mesh. The ranger's screen shows exactly what has arrived, and says "delivered" only after the ranger's phone has confirmed every byte.**

**Track:** CHRP, Off-Grid Everything (Off-Grid Mesh)

## What it does and why
When someone is hurt in the backcountry, a text-only SOS leaves out a lot ("which ankle? can you walk? where are you?"). A voice note and a photo carry that context, but LoRa gives you about 1,000 bps and 230-byte packets. Trail SOS:

1. **Squeezes the media.**
   - Voice is encoded with **Codec2 at 1200 bps**: 5 s becomes **750 B**.
   - The photo is sent in layers: an **AI caption** first (≤150 B, one packet), then a **48×48 grayscale WebP thumbnail** (380 B).
   - An **AI transcript** of the voice note goes first of all.
2. **Sends the most useful part first:** transcript → caption → audio → thumbnail. The ranger can read the situation within the first couple of packets.
3. **Survives a lossy mesh:**
   - hop-by-hop ARQ: 3 retries, then the alternate route through E
   - one XOR parity chunk per 4 data chunks, which rebuilds a single lost chunk without a resend
   - a destination NACK for anything parity can't fix
   - a source probe when nothing comes back
4. **Is honest about status:** `queued → in_transit (hop n of m) → partial (lists the parts actually reassembled) → delivered` (only after the end-to-end ack, whose CRC32 covers the whole message) or `failed` (with a reason).

## Accuracy and uncertainty
- **Delivered means delivered.** The state is set by exactly one event, `e2e_ack_received`. The ranger's phone sends that ack only after every part has been reassembled and has passed CRC32. The ack carries a CRC32 of the whole message, and the hiker's phone checks it against what it sent. Test 12 checks this "if and only if" rule over 290 trials. Another test feeds the tracker every "looks done" event except the ack and confirms it is not marked delivered.
- **Partial lists only what is really there.** The parts list is rebuilt from `reassembled` events after every single event (test 13).
- **The ETA comes from what has been observed:** remaining bytes ÷ effective rate. The effective rate uses the per-frame loss seen so far (Laplace-smoothed), so the ETA is recalculated after every loss. It is never negative, shows 0 only once delivered, and shows nothing for `failed` (test 14).
- **AI output is labeled.** Each AI result carries a confidence. The transcript and caption carry it inside the packet (1 byte), so the *ranger's* phone applies the label itself. Anything below `AI_CONFIDENCE_THRESHOLD` (0.7) is shown as **"unverified"**. A sketch rebuilt from the caption is always stamped **"AI reconstruction, not the real photo"** and inherits the caption's confidence (test 15).
- **Corrupted frames are rejected.** CRC32 covers the header and the payload. Half of the simulated losses are delivered as bit-flipped frames, which the receiver rejects.

## Tools used
- **NicrronNinoo** (AI provider, sponsor): `ai/provider.py` defines the `caption_image / transcribe / rebuild_image` interface.
  - The NicrronNinoo adapter is wired to `NICRRONNINOO_*` env vars. Its request format is a placeholder until the playground details (base URL, auth, model, sample request) are filled in.
  - Until then, or on any API error, the app falls back to the **mock provider** and logs that it did.
- **Block Convey PRISM** (observability, sponsor): `observability/prism.py`.
  - Sends one trace per AI call with `POST /api/traces`, authenticated with the `X-PRISMtrace-Key` header.
  - `session_id` = SOS message ID and `agent_id` = `trail-sos`.
  - `metadata` holds the network facts: bytes sent, packets, resends, hops, route, parity recoveries, final delivery state and AI confidence.
  - Traces are sent from a background worker once the message settles, so tracing never slows the mesh.
  - At startup the app calls `POST /api/setup-doctor/handshake`.
  - With no keys it logs **"tracing disabled"** and runs normally.
- Codec2 (`pycodec2`), Pillow (WebP), FastAPI + WebSocket, pytest.

## How to run
```powershell
# one-time
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env      # optional: add PRISM / NicrronNinoo keys

# app  ->  http://127.0.0.1:8000
.\run.ps1                    # or: make run
# tests / benchmark
.\run.ps1 test               # or: make test
.\run.ps1 bench              # or: make bench
```

**Demo controls on the page:**
- **Send SOS** sends the bundled sample voice clip and sample photo.
- Sliders set the loss % and the sim speed; a dropdown picks the data-rate preset (1,000 bps or ~22,000 bps).
- **"Knock out node mid-send"** removes the chosen node once the ranger has received 4 chunks. **Knock out now** and **Restore all** do it by hand.
- Seeds make every run repeatable.

## Design notes
- **Packet:** a 19-byte header (version, msg id, chunk index, total, content type plus parity flag, parts mask, src, dst, hop, retransmission generation, CRC32) and a payload of at most 200 B. That makes 219 B at most, under the **230 B** limit, which `encode()` enforces.
- **Airtime** = bytes × 8 / rate, per hop: 230 B at 1,000 bps takes 1.84 s.
- **Topology:** Hiker(0,0) → B(1,0) → C(2,0) → D(3,0) → Ranger(4,0), with E at (2,1). The range is 1.6 km. Routes are Dijkstra by distance, so the primary path goes through C and the alternate goes B→E→D.
- **Randomness:** all of it comes from a single seeded `random.Random`.

## Test results
`pytest`: **55 passed, 2 skipped.** The 2 skipped tests are the live NicrronNinoo and live PRISM checks, which run automatically when credentials are in `.env`.

| # | Test | Result |
|---|---|---|
| 1–2 | ≤230 B packets, oversized payload raises, exact round-trip, every flipped byte rejected | pass |
| 3–4 | chunking with reorder and duplicates; parity rebuilds every 1-missing combination, reports failure for 2 missing | pass |
| 5 | 230 B @ 1,000 bps = 1.84 s | pass |
| 6 | voice: Codec2 1200 gives 751 B for 5 s (750 B of Codec2 plus 1 B codec id); thumbnail 380 B; caption ≤150 B | pass |
| 7 | 10,000 frames at 20% loss → within 20% ± 1.5% | pass |
| **8** | **200 seeded trials, 3 hops, 20% loss → 200/200 (100%) byte-identical and e2e-acked. Average 21.7 resends; median delivery 42.2 s (1,000 bps)** | **pass** |
| 9 | C knocked out mid-send (30 seeds) → reroutes through E and is delivered byte-identical | pass |
| 10 | Ranger or D dead → `failed` with "No end-to-end ack … unreachable / no route", never delivered | pass |
| 11 | transcript and caption complete before audio and thumbnail (200/200 at 20% loss) | pass |
| 12–15 | delivered ⇔ e2e-ack; partial lists exactly what was reassembled; ETA → 0 and reacts to loss; AI labels | pass |
| 16 | mock AI provider, end to end (transcript, caption, rebuild, decoded audio) | pass |
| 17 | PRISM against a mocked HTTP server: 3 traces, `X-PRISMtrace-Key`, required fields, `session_id` = msg id, network metadata; handshake OK / 401 | pass |
| 18 | no PRISM keys → "tracing disabled"; the app still delivers end to end | pass |
| 19 | FastAPI: POST `/api/sos` → WebSocket stream → `delivered`; media endpoints; knockout reroute via the API | pass |
| 20 | Playwright UI smoke test | not built (optional) |

## Benchmark (`BENCHMARK.md`, 100 seeded trials per row, 4-hop demo mesh)
| Rate preset | Loss/link | Bytes on air | Useful media bytes | Useful/air | Air bytes per voice-second delivered | Resends | Parity recoveries | NACKs | Delivery rate | Median delivery (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| long_range (1,000 bps) | 0% | 9,935 | 1,343 | 14% | 1,987 | 0.0 | 0.00 | 0.00 | 100.0% | 25.3 |
| long_range (1,000 bps) | 10% | 12,279 | 1,343 | 11% | 2,456 | 12.9 | 0.00 | 0.00 | 100.0% | 36.6 |
| long_range (1,000 bps) | 20% | 15,541 | 1,343 | 9% | 3,108 | 30.6 | 0.06 | 0.01 | 100.0% | 47.9 |
| long_range (1,000 bps) | 30% | 19,130 | 1,343 | 7% | 3,826 | 51.8 | 0.26 | 0.02 | 100.0% | 60.5 |
| fast (21,875 bps) | 0% | 9,935 | 1,343 | 14% | 1,987 | 0.0 | 0.00 | 0.00 | 100.0% | 1.2 |
| fast (21,875 bps) | 10% | 12,219 | 1,343 | 11% | 2,444 | 12.5 | 0.00 | 0.00 | 100.0% | 3.2 |
| fast (21,875 bps) | 20% | 15,506 | 1,343 | 9% | 3,101 | 30.8 | 0.05 | 0.01 | 100.0% | 5.6 |
| fast (21,875 bps) | 30% | 19,343 | 1,343 | 7% | 3,869 | 52.7 | 0.30 | 0.06 | 100.0% | 8.1 |

**How to read it:**
- A 5-second voice note, a caption, a transcript and a real thumbnail fit in **1,343 B** of useful payload. The voice alone costs **150 B per second**.
- "Bytes on air" counts every hop (×4), every retry, every hop ACK and the parity overhead. That is why useful/air is about 14% at 0% loss, or about 54% per hop.
- Parity overhead is heaviest on the one-chunk parts (transcript and caption), where the parity chunk is effectively a second copy. Those parts get the most protection on purpose.

## Known limitations
- **The radio model is simplified.** There are no collisions or CSMA. ACKs are sent immediately and don't queue behind data. Routing uses global topology knowledge plus each node's own list of suspect neighbours.
- **The AI runs in the cloud in this demo.** A hiker with no signal can't reach a cloud API, so a real deployment would run captioning and transcription on the device. The demo uses the provider API (or the mock) in its place.
- **NicrronNinoo adapter not finished:** the request format waits for the API details. The mock provider is used until they are added.
- **Codec2 is used directly; no fallback was needed.** `pycodec2` installed fine. The Opus fallback (ffmpeg from `imageio-ffmpeg`) is implemented but was not exercised on this machine.
- **The sample data is synthetic.** The voice is Windows text-to-speech and the photo was drawn with Pillow (`scripts/make_samples.py`).
- **Delivery times are simulated seconds.** The UI replays them at an adjustable speed.
- **No Playwright UI test** (#20); the UI is covered by the API and WebSocket smoke test.
