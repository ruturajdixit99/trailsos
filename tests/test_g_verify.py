"""G. Sent-vs-received comparison and ranger-side verification."""
from PIL import Image

from ai.provider import MockProvider
from media import image as mimage
from media.verify import MISMATCH, VERIFIED, compare
from server.sos import RangerView, build_sos, sent_vs_received
from tests.helpers import run_trial


def test_compare_tolerates_spelling_flags_real_changes():
    assert compare("North Ridge trail, mile four", "Northridge Trail, Mile 4")["verdict"] == VERIFIED
    c = compare("I hurt my ankle on North Ridge", "I hurt my knee on South Ridge")
    assert c["verdict"] != VERIFIED and c["substitutions"] == 2  # never "verified" when key words changed
    assert {o["sent"] for o in c["ops"] if o["op"] == "replace"} >= {"ankle"}


def test_image_compare_detects_changes():
    t, _ = mimage.make_thumbnail(Image.open("samples/trail_photo.jpg"))
    assert mimage.compare_sent_received(t, t)["identical"]
    other, _ = mimage.make_thumbnail(Image.new("RGB", (64, 64), (10, 10, 10)))
    c = mimage.compare_sent_received(t, other)
    assert not c["identical"] and c["same_pixels_pct"] < 50


class _DifferentEars(MockProvider):
    """A ranger whose speech model hears something else -> verification must flag it."""
    def transcribe(self, audio_wav):
        return "The weather is fine, no problem here.", 0.9

    def check_caption(self, image, caption):
        return 0.1, "Caption does not match: no person visible."


def _ranger(provider, seed=2):
    p = build_sos(MockProvider())
    t = run_trial(p.parts, seed=seed)
    rv = RangerView(provider)
    for part, data in t.sim.rx_parts[t.msg_id].items():
        rv.update(part, data, rebuild=False)
    return p, t, rv


def test_verification_agrees_when_content_matches():
    p, t, rv = _ranger(MockProvider())
    assert rv.verify_transcript() is not None and rv.verify_caption() is not None
    assert rv.view["verify_transcript"]["verdict"] == VERIFIED
    assert rv.view["verify_caption"]["verdict"] == "consistent"
    rows = {r["part"]: r for r in sent_vs_received(p, t.sim.rx_parts[t.msg_id], rv)}
    assert all(r["status"] == "identical" and r["sent_crc"] == r["received_crc"] for r in rows.values())
    assert rows["thumbnail"]["network"]["identical"] and rows["thumbnail"]["compression"]["psnr_db"] > 30
    assert rows["transcript"]["diff"]["similarity"] == 1.0


def test_verification_flags_mismatch():
    _, _, rv = _ranger(_DifferentEars())
    rv.verify_transcript()
    rv.verify_caption()
    assert rv.view["verify_transcript"]["verdict"] == MISMATCH
    assert rv.view["verify_caption"]["verdict"] == "inconsistent"


def test_sent_vs_received_reports_missing_parts():
    p = build_sos(MockProvider())
    rows = sent_vs_received(p, {}, RangerView(MockProvider()))
    assert all(r["status"] == "not yet received" and r["received_crc"] is None for r in rows)
