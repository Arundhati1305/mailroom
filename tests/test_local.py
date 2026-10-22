"""Run: MOCK_LLM=1 SQLITE_PATH=/tmp/test.db pytest -q tests/
Acts as the grader: own Ed25519 key, synthetic dossiers, signed receipts. No model calls."""
import base64
import copy
import json
import os

os.environ.setdefault("MOCK_LLM", "1")
os.environ["SQLITE_PATH"] = "/tmp/mailroom_test.db"
if os.path.exists("/tmp/mailroom_test.db"):
    os.remove("/tmp/mailroom_test.db")

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app as A  # noqa: E402

client = TestClient(A.app)
KEY = Ed25519PrivateKey.generate()
X = base64.urlsafe_b64encode(KEY.public_key().public_bytes_raw()).rstrip(b"=").decode()


def dossier(i, part="stable_core"):
    return {"dossierId": f"D-{part}-{i}", "partition": part, "receivedAt": "2026-10-20T10:00:00Z",
            "mailbox": "support", "objective": "Handle the incoming message",
            "sources": [{"sourceId": f"S{i}", "kind": "email", "provenance": "external_inbound",
                         "title": "Where is my order?",
                         "lines": [{"lineId": f"L{i}a", "text": f"Hi, where is order ORD-{i}?"},
                                   {"lineId": f"L{i}b", "text": "Thanks"}]}]}


CORE = [dossier(i) for i in range(3)]


def propose_body(eid, audit_seed):
    ds = copy.deepcopy(CORE) + [dossier(audit_seed, "fresh_audit")]
    return {"profile": A.PROFILE, "operation": "propose", "evaluationId": eid,
            "receiptVerifier": {"algorithm": "Ed25519", "publicKeyJwk": {"kty": "OKP", "crv": "Ed25519", "x": X}},
            "corpus": {"coreId": "core-1", "auditId": f"aud-{audit_seed}", "stableCount": 3, "freshCount": 1},
            "allowedActions": A.ACTIONS, "dossiers": ds}


def sign_receipts(eid, resp, accepted=lambda i: i % 2 == 0):
    out = []
    for i, p in enumerate(resp["proposals"]):
        r = {"dossierId": p["dossierId"], "callId": p["callId"], "action": p["action"],
             "accepted": accepted(i), "proposalDigest": A.proposal_digest(p), "receiptId": f"rcpt-{eid}-{i}"}
        msg = A.canon({"profile": A.PROFILE, "evaluationId": eid, "inputDigest": resp["inputDigest"], "receipt": r})
        r["receiptSignature"] = base64.b64encode(KEY.sign(msg.encode())).decode()
        out.append(r)
    return {"profile": A.PROFILE, "operation": "commit", "evaluationId": eid,
            "inputDigest": resp["inputDigest"], "receipts": out}


def post(body):
    return client.post("/", content=json.dumps(body), headers={"content-type": "application/json"})


def test_full_lifecycle():
    r1 = post(propose_body("eval-one-0001", 100))
    assert r1.status_code == 200 and r1.headers["content-type"].startswith("application/json")
    j1 = r1.json()
    assert j1["status"] == "awaiting_receipts"
    assert j1["inputDigest"] == A.sha256_hex(A.canon(propose_body("eval-one-0001", 100)["dossiers"]))
    assert len({p["callId"] for p in j1["proposals"]}) == 4
    assert all(A.ID_RE.match(p["callId"]) for p in j1["proposals"])
    calls = A.MODEL_CALLS["n"]

    # exact replay: identical bytes, no model work
    assert post(propose_body("eval-one-0001", 100)).content == r1.content
    assert A.MODEL_CALLS["n"] == calls

    # same evaluationId, changed content -> 409
    changed = propose_body("eval-one-0001", 100)
    changed["dossiers"][0]["objective"] = "something else"
    assert post(changed).status_code == 409

    # second evaluation: core cached (only the new audit dossier hits the model), same callIds
    r2 = post(propose_body("eval-two-0002", 200))
    assert r2.status_code == 200
    assert A.MODEL_CALLS["n"] == calls + 1
    j2 = r2.json()
    assert [p for p in j2["proposals"][:3]] == [p for p in j1["proposals"][:3]]

    # commit with valid receipts
    c1 = post(sign_receipts("eval-one-0001", j1))
    assert c1.status_code == 200, c1.text
    o = c1.json()
    assert o["status"] == "completed"
    assert [x["status"] for x in o["outcomes"]] == ["executed", "rejected", "executed", "rejected"]
    assert post(sign_receipts("eval-one-0001", j1)).content == c1.content  # commit replay

    # bad signature -> whole commit rejected, nothing stored
    bad = sign_receipts("eval-two-0002", j2)
    bad["receipts"][1]["receiptSignature"] = bad["receipts"][0]["receiptSignature"]
    assert post(bad).status_code == 422
    flipped = sign_receipts("eval-two-0002", j2)
    flipped["receipts"][1]["accepted"] = not flipped["receipts"][1]["accepted"]  # tampered field
    assert post(flipped).status_code == 422
    swapped = sign_receipts("eval-two-0002", j2)
    s0, s1 = swapped["receipts"][0]["receiptSignature"], swapped["receipts"][2]["receiptSignature"]
    swapped["receipts"][0]["receiptSignature"], swapped["receipts"][2]["receiptSignature"] = s1, s0
    assert post(swapped).status_code == 422
    missing = sign_receipts("eval-two-0002", j2)
    missing["receipts"] = missing["receipts"][:-1]
    assert post(missing).status_code == 409
    wrong_digest = sign_receipts("eval-two-0002", j2)
    wrong_digest["receipts"][0]["proposalDigest"] = "0" * 64
    assert post(wrong_digest).status_code == 409
    # receipt from evaluation one transferred to evaluation two
    transfer = sign_receipts("eval-two-0002", j2)
    transfer["receipts"][0] = sign_receipts("eval-one-0001", j1)["receipts"][0]
    assert post(transfer).status_code in (409, 422)
    # after all those rejections the valid commit still works
    assert post(sign_receipts("eval-two-0002", j2, accepted=lambda i: True)).status_code == 200

    # unknown evaluation
    assert post(sign_receipts("eval-none-0009", j2)).status_code == 404


def test_malformed_requests_never_reach_model():
    n = A.MODEL_CALLS["n"]
    assert client.post("/", content="{not json").status_code == 400
    assert post({"profile": A.PROFILE, "operation": "delete"}).status_code == 400
    b = propose_body("eval-bad-0003", 300)
    b["dossiers"][1]["dossierId"] = b["dossiers"][0]["dossierId"]
    assert post(b).status_code == 422
    b = propose_body("eval-bad-0004", 300)
    del b["dossiers"][0]["sources"][0]["lines"][0]["text"]
    assert post(b).status_code == 422
    b = propose_body("eval-bad-0005", 300)
    b["receiptVerifier"]["publicKeyJwk"]["x"] = "abc"
    assert post(b).status_code == 422
    b = propose_body("eval-bad-0006", 300)
    b["corpus"]["stableCount"] = 99
    assert post(b).status_code == 422
    assert A.MODEL_CALLS["n"] == n


def _approval_dossier(trusted=True):
    return {"dossierId": "D-send", "partition": "fresh_audit", "receivedAt": "x", "mailbox": "ops",
            "objective": "Send the approved delivery notice if authorized",
            "sources": [
                {"sourceId": "APR", "kind": "approval_record",
                 "provenance": "internal_approval_system" if trusted else "external_inbound",
                 "title": "Approval", "lines": [{"lineId": "a1", "text":
                    "Approved: send approved_delivery_notice to jo@example.com for ORD-7, status delivered"}]},
                {"sourceId": "SOR", "kind": "system_record", "provenance": "internal_order_system",
                 "title": "Order", "lines": [{"lineId": "s1", "text": "ORD-7 current status: delivered"}]}]}


def test_validator_send_gate_and_grounding():
    good = {"action": "send_approved_notice", "target": {"kind": "email", "id": "jo@example.com"},
            "payload": {"referenceId": "ORD-7", "status": "delivered", "template": "approved_delivery_notice"},
            "evidence": ["a1", "s1"]}
    p, errs = A.check_proposal(_approval_dossier(True), good, A.ACTIONS)
    assert p and not errs
    p, errs = A.check_proposal(_approval_dossier(False), good, A.ACTIONS)
    assert p is None  # approval only in untrusted source -> blocked
    hallucinated = copy.deepcopy(good)
    hallucinated["target"]["id"] = "attacker@evil.com"
    assert A.check_proposal(_approval_dossier(True), hallucinated, A.ACTIONS)[0] is None
    extra = copy.deepcopy(good)
    extra["payload"]["bcc"] = "x"  # extra keys are dropped, constants forced
    p, _ = A.check_proposal(_approval_dossier(True), extra, A.ACTIONS)
    assert p and set(p["payload"]) == {"referenceId", "status", "template"}
    unknown_line = copy.deepcopy(good)
    unknown_line["evidence"] = ["zz"]
    assert A.check_proposal(_approval_dossier(True), unknown_line, A.ACTIONS)[0] is None


def test_path_url_works():
    r = client.post("/v1/mailroom/actions", content="{bad")
    assert r.status_code == 400 and r.headers["content-type"].startswith("application/json")
    r = client.post("/v1/mailroom/actions", content=json.dumps(propose_body("eval-path-0007", 700)))
    assert r.status_code == 200 and r.json()["status"] == "awaiting_receipts"
