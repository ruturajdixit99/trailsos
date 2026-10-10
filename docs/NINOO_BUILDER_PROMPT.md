# Build Trail SOS: off-grid messaging over a simulated LoRa mesh

Build a web app called **Trail SOS** for the CHRP "Off-Grid Everything" hackathon track. Phones that have no cell service and no WiFi relay messages through each other over LoRa radio. Each LoRa packet holds about **230 bytes**, and data rates run from about **1 kbps** (long range) to about **22 kbps** (fastest Meshtastic preset). The app must send **text, voice, photos and video** across a simulated mesh of these phones and be strictly honest about what has arrived.

Everything below is a requirement unless it says "optional". When a section gives exact numbers, use them.

---

## 1. Architecture
- **Front end:** a single-page web app (TypeScript + React suggested). The whole mesh simulation runs **in the browser**, in a Web Worker if possible, so it is fast and needs no backend.
- **Server:** one small server route, used **only** to proxy AI calls to the Nicrron gateway, so the API key never reaches the browser. Read the key from a secret or environment variable named `NICRRON_API_KEY`. Never hard-code it.
- **Randomness:** every random number comes from a **seeded PRNG**, so any run can be replayed exactly.
- **Without an API key:** if the key is missing, use a **mock AI** with fixed outputs and show "AI: mock" in the header. The app must still fully work.

## 2. The mesh (simulated)
- **6 phones** on a map, in map units where 1 unit = 1.8 km by default:
  - Hiker (0,0), B (1,0), C (2,0), D (3,0), Ranger (4,0)
  - E at (2,0.7), an alternate relay between B and D
- Draw them on an SVG map. Draw links as lines colored by quality (green strong, amber marginal, red weak). Curve any link that would pass through another node, and show its numbers on hover.
- **Two radio models**, chosen in the UI:
  1. **Simple:** a fixed loss % per link (default 20%), with airtime = bytes × 8 / rate. Rate presets are 1,000 bps and 21,875 bps. A link exists if two nodes are within 1.6 map units. Half of the lost frames arrive corrupted and must be rejected by the CRC check.
  2. **Realistic LoRa:**
     - **Airtime:** the Semtech SX126x formula with a 16-symbol preamble, explicit header, CRC on, and low-data-rate optimisation when the symbol time is over 16 ms. Check it: SF7 / 125 kHz / CR 4/5 / 8-symbol preamble / 20 B payload must give **56.58 ms**.
     - **Meshtastic presets:**
       - LongSlow: SF12, 125 kHz, CR 4/8 (≈183 bps)
       - LongFast: SF11, 250 kHz, CR 4/5 (≈1,074 bps, the default)
       - MediumFast: SF9, 250 kHz (≈3,516 bps)
       - ShortFast: SF7, 250 kHz (≈10,938 bps)
       - ShortTurbo: SF7, 500 kHz (≈21,875 bps)
     - **Signal:** 915 MHz, TX 22 dBm, 2 dBi antenna at each end, log-distance path loss with exponent 3.6 (forested mountains), noise = −174 + 10·log10(BW) + 6 dB, and a UI slider for extra terrain loss in dB. Add per-link shadowing (Gaussian, σ 3 dB, fixed per seed, the same in both directions) and per-packet fading (σ 3 dB).
     - **Demodulation:** a frame survives only if its instantaneous SNR is at or above the floor for its spreading factor: SF7 −7.5, SF8 −10, SF9 −12.5, SF10 −15, SF11 −17.5, SF12 −20 dB. A link is usable if its mean SNR is within 3 dB of that floor.
     - **Collisions:** overlapping frames at a receiver collide unless one is at least 6 dB stronger (capture effect). Radios are **half-duplex**: a node can't receive while it's transmitting.
     - **Listen-before-talk** with random backoff, and random jitter before every retry.

## 3. Packet format (hard limit 230 bytes)
- **Header, 19 bytes:**
  - version (1)
  - message id (4)
  - chunk index (2)
  - total chunks in this part (2)
  - content type (1; the high bit marks a parity chunk)
  - mask of the parts in this message (1)
  - source (1)
  - destination (1)
  - hop count (1)
  - retransmission generation (1)
  - CRC32 over header and payload (4)
- **Payload:** at most 200 bytes, so the largest packet is **219 B**. Encoding must **throw** if a packet would exceed 230 B. Decoding must **reject** any packet whose CRC doesn't match.
- **Content types:** text/transcript, caption/description, audio, thumbnail, video, hop-ACK, end-to-end ACK, NACK.

## 4. Getting a message through intact
1. **Chunking:** each part's bytes become a 2-byte length prefix plus the data, split into ≤200 B chunks.
2. **Parity:** add one **XOR parity chunk per group of 4** data chunks, so any single lost chunk in a group is rebuilt with no resend.
3. **Hop-by-hop ARQ:** each node sends one packet and waits for a 20 B hop-ACK. Retry up to 3 times. If all fail, mark that neighbour suspect and try an **alternate route**; if there is none, drop the packet.
4. **Routing:**
   - Use shortest path: by distance in the simple model, by **ETX** (expected transmissions, computed from each link's loss) in LoRa mode.
   - **Penalty:** each failure multiplies a neighbour's cost by (1 + 5 × failure rounds). Any ACK from that neighbour clears it.
   - **Split horizon:** a relay never forwards a packet straight back to the node it came from.
   - **Deduplication:** relays drop duplicates, keyed on (message, type, index, source, generation).
5. **Destination:** reassemble the parts and apply parity. If progress stalls, send a **NACK** listing the missing chunks. NACKs back off exponentially, and the limit counts only NACKs sent since the last new packet arrived.
6. **Source:** if nothing comes back, **probe** by resending one chunk, up to 3 times in the simple model and 8 in LoRa, then mark the message **failed** with a clear reason.
7. **End-to-end ACK:** the destination sends **one** end-to-end ACK, carrying a CRC32 of the whole message, only when every part is reassembled and verified. The source checks that CRC.

## 5. Squeezing the media (the byte budgets matter most)
| Content | How | Target size |
|---|---|---|
| Text | UTF-8, plus 1 confidence byte (100 = typed by a person) | ≤197 B (1 packet) |
| Voice, 5 s | **Codec2 1200 bps** via WebAssembly if available (5 s = 750 B). Otherwise Opus at its lowest bitrate via WebCodecs, labelled as a fallback in the UI. Also send a **Whisper transcript**. | ≈750 B + transcript |
| Photo | AI caption (≤150 B including 1 confidence byte), plus a **48×48 grayscale WebP** thumbnail at the best quality that fits | ≤800 B (we get 380 B) |
| Video, 5 s | Downscale every frame to 48×48 grayscale. Keep the first frame, every frame whose mean absolute change from the last kept frame is ≥6/255, and the last frame. Cap at 6, then thin to fit. Encode at their real timestamps (one animated WebP, or one tiny WebP per frame). Also send an AI **description** of a middle frame. | ≤1,600 B (we get 4 frames in 1,292 B) |
| SOS | voice + photo in one message | ≈1.3 KB, 12 packets |

**Send priority:** transcript/text → caption → audio → thumbnail → video, so the useful text arrives first.

**Captions must fit honestly:**
- Ask the AI for at most 110 characters, and ask it once to shorten if the reply is still too long.
- As a last resort, cut at a word boundary and add a visible "…".
- **Never cut mid-word.** The sender's screen shows exactly the text that goes on air.

**Sample media:** bundle a 5 s spoken voice clip, a trail photo of a hiker with an injured ankle, and a 5 s video of the hiker waving. All of it must be fake or sample data.

## 6. AI, all through the Nicrron gateway (OpenAI-compatible)
- **Connection:** base URL `https://api.nicrron.ai/v1`, with header `Authorization: Bearer <NICRRON_API_KEY>`. Call it from the server route only.
- **Transcribe:** `POST /audio/transcriptions` with model `openai/whisper-1` and `response_format=verbose_json`.
  - Confidence = exp(duration-weighted mean `avg_logprob`) × (1 − mean `no_speech_prob`).
  - Add 1 s of trailing silence to the audio sent for transcription only.
- **Caption / describe:** `POST /chat/completions` with model `anthropic/claude-haiku-4-5`, the image as a data URL, and `temperature` 0. Ask for JSON `{caption, confidence}` and cap the confidence at 0.95.
- **Rebuild sketch (receiver):** `POST /images/generations` with model `google/nano-banana` and only `prompt` (plus an optional `seed`).
  - It returns a job: poll `GET /images/generations/{id}` until `succeeded`, then download `GET /images/generations/{id}/content`.
  - Downscale the result and stamp it **"AI RECONSTRUCTION, NOT THE REAL PHOTO"**.
  - It costs about $0.04, so run it automatically for SOS only and on a button press for other photos.
- **Labels:** any AI result below 0.7 confidence is labelled **"unverified"**, and every rebuilt image is labelled **"AI reconstruction"**.
- **Failures:** if any AI call fails, fall back to the mock for that call and say so.

## 7. Honest status (the property judges care about most)
- **States per message:** `queued → in transit (hop n of m) → partial (lists exactly which parts are reassembled, e.g. "audio 3/4") → delivered ✓✓ | failed (with reason)`.
- **"Delivered"** is set by **one event only**: the source received the end-to-end ACK and its CRC matched.
- **ETA** = remaining chunk airtime ÷ (1 − observed loss)², plus the pipeline across the remaining hops. Recompute it on every loss. It is never negative, 0 only once delivered, and blank when failed.
- **Route:** show the route actually taken (from the packets that arrived), and say "planned" until packets have arrived.

## 8. Screens

### Home: live mesh
- **Two big iPhone-style phones:** the sender (left) and the receiver (right) of the **focused** message.
  - The sender shows the original media, the on-device AI results with confidence and model, and the exact payload bytes.
  - The receiver shows the status badge, ETA, route, resends, each part as it arrives (with progress bars), the received audio player, the thumbnail, the video flipbook, and the labelled AI sketch.
- **Controls:**
  - radio model and its settings: loss %, rate, preset, hop km, terrain dB
  - sim speed and seed
  - knock out a relay (now, or automatically mid-send), and restore all
- **Mesh map:** packet dots moving along links (data blue, parity violet, ACK green), "✗ cause" where a frame is lost, queue counts on each node, and the route in use highlighted. The stats strip shows bytes on air, resends, observed loss, parity rebuilds, reroutes and effective bps.
- **Phones in the mesh (multi-device testing):**
  - a card for **each of the 6 phones**: online/offline, frames tx / rx / relayed, and a send row (recipient, kind = text / voice / photo / video / SOS, a text box for text, a Send button)
  - an **inbox** and a **sent** list with status chips
  - **any phone can send to any phone; many messages can be in flight at once on the same shared radios**
- **One-click scenarios:**
  - "Everyone reports to Ranger" (5 messages)
  - "Ranger broadcasts to all" (5 separate confirmed deliveries)
  - "Two-way chat" (4)
  - "Busy mesh" (6 phones in different directions at once, staggered by 0.5 s)
- **All messages table:** #, from → to, what, size, status, sent at, time, route, verification badges. Clicking a row or inbox item **focuses** that message, and every detailed panel follows it.
- **Run summary** for the focused message, in plain language with a Copy button: result, route planned vs used, knockouts and reroutes, when each part arrived, sent vs received, loss causes, busiest and worst link, recovery work, efficiency.
- **Sent vs received + verification** for the focused message:
  - Per part: sent vs received byte count and CRC32, with "identical ✓" or "DIFFERENT ✗".
  - **Text:** word-level diff, deletions red and struck through, insertions green.
  - **Transcript check:** the receiver re-transcribes the audio it received and diffs it against the transcript. Normalise case, punctuation and numbers ("4" = "four"), and merge compounds ("North Ridge" = "Northridge"). Show word match %, WER and a verdict: verified (≥85%), partly matches (≥60%), mismatch.
  - **Caption / description check:** the vision model scores how well the caption describes the received thumbnail or video frame, as a JSON score from 0–1 with a note.
  - **Image:** original photo, sent thumbnail, received thumbnail, a sent-vs-received pixel-difference map (white = identical), the compression difference vs the original, and PSNR.
  - **Video:** the original clip, the sent keyframes, the received flipbook, frames identical x/y, timing identical.
  - **Audio:** the original recording and the received decoded audio, side by side.
- **Network ledger** for the focused message:
  - totals: sender frames and bytes, receiver unique chunks, duplicates discarded, frames delivered/lost, loss causes as a coloured bar
  - a **per-link table:** sent, received, lost by cause (fade/random, collision, half-duplex, CRC, node down), ACKs sent/lost, mean SNR
  - a **timeline:** one row per chunk with sends, losses, arrivals and parity rebuilds
  - a **per-chunk table:** source sends, frames, losses, path, and per-hop "got through / tries", e.g. `H›B 3/4`
- **Event log:** every frame, with the message number, who → who, the event, and the loss cause with RSSI/SNR.

### ✓ Quick check (one button, finishes in a few seconds)
Run these as real tests on the real code. Use the mock AI so results are repeatable and free. Show ✓/✗ for each, with its evidence:
1. 5+ nodes; the Hiker → Ranger route has at least 3 hops; an alternate path through E exists.
2. Every packet of every content kind is ≤230 B; report the largest.
3. Voice ≤800 B, image ≤800 B, video ≤1,600 B; report the ratios against the original WAV, JPEG and MP4.
4. The voice + image SOS over **3 hops at 20% loss: ≥99% of 100 seeded runs arrive byte-identical** (we get 100/100).
5. **Zero** "delivered" without an end-to-end ACK across all runs, including 15 runs with relay D dead; those must end "failed" with a reason.
6. With relay C knocked out mid-send, at least 18/20 runs are delivered via E.
7. 6 phones sending at once in different directions (8 seeds): every message intact.
8. LoRa airtime matches Semtech; the SOS on LongFast over 1.8 km hops is delivered in at least 9/10 runs.
9. Media per byte: useful bytes per SOS, and bytes on air across 3 hops at 20% loss.

### Transfer Lab (second page)
- **Inputs:** content variants (text, Codec2 audio, thumbnail, full SOS, plus uncompressed G.711 audio at 40 KB and the original 15 KB photo for comparison), radio settings, relay failure, trials (default 30), seed.
- **Method:** send each variant alone, many times, in a background worker with a progress bar.
- **Charts:**
  - **Speed:** time until the receiver has everything (median dot, line from the fastest 10% to the slowest 10%, log/linear toggle), and how much has arrived over time (average progress curve).
  - **Loss:** frames lost on the air, share delivered intact, why frames were lost (shaded table), and bytes on air per useful byte.
  - **Sweep:** vary loss %, hop distance or terrain dB, plotting median time and delivered-intact % for each variant.
- **Tables:** an "All numbers" table and an "Every trial" table (sortable), with CSV and JSON download.
- **Style:** colourblind-safe categorical colours in a fixed order, light and dark mode, and a legend plus a table view for every chart.

## 9. Acceptance numbers (the build is done when these hold)
- Largest packet **219 B**. Voice **≈751 B** per 5 s. Thumbnail **≈380 B**. Video **≈1.3 KB**. SOS **12 packets**.
- **100/100** byte-identical at 20% loss over 3 hops.
- **0** false "delivered".
- At 20% loss and 1 kbps, median time to have everything: text about **4 s**, thumbnail about **15 s**, Codec2 voice about **22 s**, SOS about **40 s**. Uncompressed 40 KB audio takes about **14 min**, and on realistic LoRa it **never arrives**: the Lab must show that honestly.
- The Quick check shows **9/9**.

## 10. Known limits to state in the README
- The AI is cloud-hosted in the demo, but would run on the phone in reality.
- Hop ACKs don't use listen-before-talk.
- There is no duty-cycle limit (US915).
- Terrain is modelled as a single exponent plus a dB slider.
- The sample media is synthetic.
