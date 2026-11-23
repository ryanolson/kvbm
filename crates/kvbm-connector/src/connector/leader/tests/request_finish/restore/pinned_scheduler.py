# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive the live Rust connector with pinned vLLM scheduler callbacks."""

import __future__
import ast
import hashlib
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace


class RequestStatus:
    PREEMPTED = "PREEMPTED"
    RUNNING = "RUNNING"
    WAITING_FOR_STREAMING_REQ = "WAITING_FOR_STREAMING_REQ"
    WAITING_FOR_REMOTE_KVS = "WAITING_FOR_REMOTE_KVS"
    FINISHED_ABORTED = "FINISHED_ABORTED"
    FINISHED_STOPPED = "FINISHED_STOPPED"

    @staticmethod
    def is_finished(status):
        return status in (RequestStatus.FINISHED_ABORTED, RequestStatus.FINISHED_STOPPED)

class Request(SimpleNamespace):
    __hash__ = object.__hash__

def read():
    return json.loads(sys.stdin.readline())


def reply(**result):
    print(json.dumps(result), flush=True)


def finish_connector(request):
    assert request.is_finished()
    reply(callback="request_finished", request=request.request_id)
    return read()["pending"], None


def update_connector(output):
    reply(callback="update_connector_output", output=vars(output))
    assert read() == {"updated": True}


def schedule_test(methods, request_id, waiting_for_remote):
    frees = []
    status = RequestStatus.WAITING_FOR_REMOTE_KVS if waiting_for_remote else RequestStatus.PREEMPTED
    request = Request(request_id=request_id, status=status, get_finished_reason=lambda: "stop")
    request.is_finished = lambda: RequestStatus.is_finished(request.status)
    waiting = [request]

    def remove_requests(requests):
        for removed in requests:
            assert removed is request
            waiting.remove(removed)

    scheduler = type("PinnedScheduler", (), methods)()
    scheduler._connector_finished = finish_connector
    scheduler._get_routed_experts = lambda _: None
    scheduler._handle_stopped_request = lambda _: True
    scheduler.connector = SimpleNamespace(update_connector_output=update_connector)
    scheduler.requests = {request_id: request}
    scheduler.waiting = SimpleNamespace(remove_requests=remove_requests)
    scheduler.kv_cache_manager = SimpleNamespace(free=lambda req: frees.append(req.request_id))
    scheduler.encoder_cache_manager = SimpleNamespace(free=lambda _: None)
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.failed_recving_kv_req_ids = set()
    return scheduler, request, frees, waiting


def load_scheduler(source):
    raw = source.read_bytes()
    # scheduler.py from vLLM 43cddb6c3765c67318ff5c9e4e7c00be2b347764.
    assert hashlib.sha256(raw).hexdigest() == (
        "f970e5d6ebcbcb3e6241ea98aed7883c7fd07d93562fa93c9da231b7546b1907"
    ), "unexpected vLLM scheduler source"
    tree = ast.parse(raw)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
    names = {"finish_requests", "_free_request", "_free_blocks", "_update_from_kv_xfer_finished"}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"RequestStatus": RequestStatus, "logger": logging.getLogger()}
    module = ast.Module(body=methods, type_ignores=[])
    exec(compile(module, str(source), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return {name: namespace[name] for name in names}, cls


def stop_preempted(source, cls, scheduler, request):
    update = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "update_from_output")
    conditions = [
        node for node in ast.walk(update)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
    ]
    settle = next(
        node for node in conditions if node.test.id == "stopped"
        and any(isinstance(child, ast.Attribute) and child.attr == "_free_request" for child in ast.walk(node))
    )
    remove = next(node for node in conditions if node.test.id == "stopped_preempted_reqs")
    # A model stop already set the terminal status. Run the original stop
    # settlement and waiting-queue removal from update_from_output.
    request.status = RequestStatus.FINISHED_STOPPED
    exec(
        compile(ast.Module(body=[settle, remove], type_ignores=[]), str(source), "exec"),
        {
            "self": scheduler,
            "request": request,
            "stopped": True,
            "status_before_stop": RequestStatus.PREEMPTED,
            "stopped_preempted_reqs": set(),
            "RequestStatus": RequestStatus,
        },
    )


def main(source):
    methods, cls = load_scheduler(source)
    first = read()
    scheduler, request, frees, waiting = schedule_test(
        methods, first["request"], first["waiting_for_remote"]
    )
    if first["command"] == "abort":
        scheduler.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED)
    else:
        assert first["command"] == "stop"
        stop_preempted(source, cls, scheduler, request)
    assert not waiting
    reply(retained=request.request_id in scheduler.requests and not frees)

    while True:
        step = read()
        if step["command"] == "end":
            reply(frees=frees)
            return
        assert step["command"] == "deliver"
        try:
            scheduler._update_from_kv_xfer_finished(SimpleNamespace(**step["output"]))
        except AssertionError as error:
            reply(assertion=error.__traceback__.tb_next.tb_lineno)
        else:
            reply(assertion=None)


main(Path(sys.argv[1]))
