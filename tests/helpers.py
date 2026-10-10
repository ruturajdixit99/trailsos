from dataclasses import dataclass, field

from mesh.packet import ContentType
from mesh.sim import MeshSim, SimConfig, demo_topology
from mesh.status import StatusTracker


@dataclass
class Trial:
    sim: MeshSim
    msg_id: int
    tracker: StatusTracker
    events: list = field(default_factory=list)

    @property
    def state(self):
        return self.tracker.state

    def kinds(self, kind):
        return [e for e in self.sim.events if e.kind == kind and e.msg_id in (self.msg_id, None)]

    def identical(self, parts) -> bool:
        got = self.sim.rx_parts.get(self.msg_id, {})
        return all(got.get(ContentType(ct).name.lower()) == b for ct, b in parts.items())


def run_trial(parts, seed=0, loss=0.2, topology=None, rate=1000.0, knockout=None,
              knockout_after_rx=4, dead_from_start=(), src="Hiker", dst="Ranger", max_t=1e5) -> Trial:
    sim = MeshSim(topology or demo_topology(), SimConfig(rate_bps=rate, loss=loss, seed=seed))
    for n in dead_from_start:
        sim.set_alive(n, False)
    msg_id = sim.send_message(src, dst, parts)
    tr = StatusTracker(msg_id, rate)
    for ev in sim.events:
        tr.on_event(ev)
    sim.listeners.append(tr.on_event)
    if knockout:
        rx = {"n": 0}

        def watch(ev):
            if ev.kind == "received" and ev.msg_id == msg_id:
                rx["n"] += 1
                if rx["n"] == knockout_after_rx:
                    sim.schedule(0.0, sim.set_alive, knockout, False)
        sim.listeners.append(watch)
    sim.run(max_t=max_t)
    return Trial(sim, msg_id, tr)
