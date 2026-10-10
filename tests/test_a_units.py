"""A. Unit tests: packets, chunking/parity, airtime, media budgets."""
import itertools
import os
import random

import pytest

from media import chunking, voice
from media.text import CAPTION_BUDGET
from mesh.packet import (
    HEADER_SIZE, MAX_PACKET, MAX_PAYLOAD, PARITY_FLAG, ChecksumError, ContentType, Packet, PacketError, airtime_s,
)


# 1 ---------------------------------------------------------------------------
def test_header_within_budget():
    assert HEADER_SIZE <= 30


@pytest.mark.parametrize("n", [0, 1, 100, MAX_PAYLOAD])
def test_every_encoded_packet_fits(n):
    p = Packet(1, 0, 1, ContentType.AUDIO, 0, 4, os.urandom(n))
    assert len(p.encode()) <= MAX_PACKET


def test_all_sos_packets_fit(sos_payload):
    from mesh.sim import MeshSim
    sim = MeshSim()
    mid = sim.send_message("Hiker", "Ranger", sos_payload.parts)
    for p in sim.tx[mid].packets.values():
        assert len(p.encode()) <= MAX_PACKET


def test_oversized_payload_raises():
    with pytest.raises(PacketError):
        Packet(1, 0, 1, ContentType.AUDIO, 0, 4, b"x" * (MAX_PAYLOAD + 1)).encode()


def test_oversized_raw_rejected_on_decode():
    with pytest.raises(PacketError):
        Packet.decode(b"\x01" * (MAX_PACKET + 1))


# 2 ---------------------------------------------------------------------------
def test_roundtrip_exact():
    rng = random.Random(1)
    for _ in range(200):
        p = Packet(rng.getrandbits(32), rng.randrange(65536), rng.randrange(65536),
                   rng.choice([1, 2, 3, 4, 5, 6, 7, 3 | PARITY_FLAG]), rng.randrange(256), rng.randrange(256),
                   bytes(rng.getrandbits(8) for _ in range(rng.randrange(MAX_PAYLOAD + 1))),
                   hop=rng.randrange(256), parts_mask=rng.randrange(256), gen=rng.randrange(256))
        assert Packet.decode(p.encode()) == p


def test_any_flipped_byte_is_rejected():
    raw = Packet(0xDEADBEEF, 3, 9, ContentType.THUMBNAIL, 0, 4, os.urandom(150), hop=2).encode()
    for i in range(len(raw)):
        bad = bytearray(raw)
        bad[i] ^= 0x5A
        with pytest.raises(PacketError):  # ChecksumError (or version error, also a PacketError)
            Packet.decode(bytes(bad))


def test_flipped_payload_byte_is_checksum_error():
    raw = bytearray(Packet(7, 0, 1, ContentType.CAPTION, 0, 4, b"hello world").encode())
    raw[-1] ^= 1
    with pytest.raises(ChecksumError):
        Packet.decode(bytes(raw))


# 3 ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", [0, 1, 197, 198, 199, 200, 750, 801, 4000])
def test_chunk_reassemble_identity(size):
    data = os.urandom(size)
    chunks = chunking.split(data)
    assert all(len(c) <= MAX_PAYLOAD for c in chunks)
    a = chunking.PartAssembler(len(chunks))
    for i, c in enumerate(chunks):
        a.add_data(i, c)
    assert a.assemble() == data


def test_out_of_order_and_duplicates():
    data = os.urandom(1234)
    chunks = chunking.split(data)
    order = list(range(len(chunks))) * 3
    random.Random(5).shuffle(order)
    a = chunking.PartAssembler(len(chunks))
    for i in order:
        a.add_data(i, chunks[i])
    assert a.complete and a.assemble() == data


# 4 ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", [100, 750, 1600, 2500])
def test_parity_rebuilds_any_single_missing_per_group(size):
    data = os.urandom(size)
    chunks = chunking.split(data)
    par = chunking.parity_chunks(chunks)
    groups = [list(range(g * 4, min(g * 4 + 4, len(chunks)))) for g in range(len(par))]
    # drop exactly one chunk in EVERY group, for every combination
    for drop in itertools.product(*groups):
        a = chunking.PartAssembler(len(chunks))
        for i, c in enumerate(chunks):
            if i not in drop:
                a.add_data(i, c)
        for g, p in enumerate(par):
            a.add_parity(g, p)
        assert sorted(a.try_recover()) == sorted(drop)
        assert a.assemble() == data


def test_parity_reports_failure_with_two_missing():
    data = os.urandom(800)
    chunks = chunking.split(data)          # 5 chunks -> groups [0..3], [4]
    par = chunking.parity_chunks(chunks)
    for drop in itertools.combinations(range(4), 2):
        a = chunking.PartAssembler(len(chunks))
        for i, c in enumerate(chunks):
            if i not in drop:
                a.add_data(i, c)
        for g, p in enumerate(par):
            a.add_parity(g, p)
        assert a.try_recover() == []
        assert not a.complete
        assert a.unrecoverable_groups() == [0]
        with pytest.raises(chunking.ReassemblyError):
            a.assemble()


# 5 ---------------------------------------------------------------------------
def test_airtime_230_bytes_at_1000bps():
    assert airtime_s(230, 1000) == pytest.approx(1.84, abs=1e-9)


def test_airtime_fast_preset():
    assert airtime_s(230, 21875) == pytest.approx(0.0841, abs=1e-3)


# 6 ---------------------------------------------------------------------------
def test_voice_budget(sos_payload):
    audio = sos_payload.parts[ContentType.AUDIO]
    print(f"\nvoice: {sos_payload.codec}, {sos_payload.voice_seconds:.1f}s -> {len(audio)} B")
    assert sos_payload.voice_seconds == pytest.approx(5.0)
    if sos_payload.codec.startswith("codec2"):
        assert len(audio) <= 800
    else:  # documented fallback: log size, budget is not enforced for Opus
        print(f"OPUS FALLBACK size: {len(audio)} B")


def test_codec2_roundtrip_length():
    if not voice.HAVE_CODEC2:
        pytest.skip("codec2 not installed (fallback in use)")
    pcm = voice.load_wav(voice.__file__.replace("media\\voice.py", "samples\\sos_voice.wav").replace(
        "media/voice.py", "samples/sos_voice.wav"))
    bits, codec = voice.encode(pcm)
    assert len(voice.decode(bits, codec)) == len(pcm)


def test_thumbnail_budget(sos_payload):
    assert len(sos_payload.parts[ContentType.THUMBNAIL]) <= 800


def test_caption_budget(sos_payload):
    assert len(sos_payload.parts[ContentType.CAPTION]) <= CAPTION_BUDGET
    from media.text import pack
    assert len(pack("é" * 500, 0.5, CAPTION_BUDGET)) <= CAPTION_BUDGET
