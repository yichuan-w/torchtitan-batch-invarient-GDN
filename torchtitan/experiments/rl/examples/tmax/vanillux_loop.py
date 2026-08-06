# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Faithful host-side port of the AI2 tmax Vanillux (mini-swe-agent) bash-only loop.

The tmax Qwen3.5-9B is SFT'd under one specific agent scaffold
(``SWERLVanilluxSandboxEnv`` in open-instruct); running it under a different
scaffold (e.g. the swe_r2e host_loop with Bash/Read/Write/Edit + an SWE system
prompt) puts the policy off-distribution and starves the solve rate. This module
reproduces that scaffold exactly, while keeping the host_loop transport: the agent
brain runs on the controller and talks to the on-box Anthropic adapter over
localhost (one packed TITO episode); each single ``bash`` action is dispatched to
the remote Daytona sandbox via ``sb.exec``.

Fidelity to ``SWERLVanilluxSandboxEnv`` / ``vanillux_solver.py``:
  * ONE ``bash`` tool -- no editor/submit tools. The agent submits by running
    ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` through bash.
  * ``system_template`` + ``instance_template`` from the vendored
    ``vanillux_prompts.py`` (mini-swe-agent v2.2.x, byte-faithful).
  * A persistent-shell wrapper preserves cwd + env across calls (``cd``/``export``
    stick), matching the env's ``_BASH_WRAPPER``.
  * Observations use mini-swe-agent head/tail truncation + a ``too_long_hint`` and
    end with ``(exit_code=N)``.
  * A format-error reminder is returned when the model emits no ``bash`` call.

Grading is unchanged and lives in the rollouter/rubric: on the submit marker this
loop returns ``submitted=True`` and the rollouter runs ``grade_tmax`` (upload
tests, ``bash /tests/test.sh``, read ``/logs/verifier/reward.txt``). A rollout that
never submits is scored 0, matching the env (tests only run on submit).
"""

from __future__ import annotations

import base64
import logging
import os
import shlex
import time
from typing import TYPE_CHECKING

from torchtitan.experiments.rl.examples.tmax.vanillux_prompts import (
    FORMAT_ERROR_TEMPLATE,
    INSTANCE_TEMPLATE,
    OBS_HEAD_CHARS,
    OBS_MAX_CHARS,
    OBS_TAIL_CHARS,
    OBS_TOO_LONG_HINT,
    SYSTEM_TEMPLATE,
)
from torchtitan.experiments.rl.harness.sandbox import Sandbox

if TYPE_CHECKING:
    from torchtitan.experiments.rl.harness.adapters.anthropic import AnthropicAdapter

logger = logging.getLogger(__name__)

# Anthropic model name the adapter answers to; arbitrary (the adapter ignores it).
ADAPTER_MODEL_NAME = "titan-actor"

# The agent submits + triggers grading by echoing this marker through bash
# (identical to open-instruct's swerl_sandbox.SUBMIT_MARKER).
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# Max bash actions before the loop stops. The tmax RL run uses --max_steps 64
# (qwen35_9b.sh); the env's VANILLUX_CALL_LIMIT=100 is only the fallback. Match 64.
_MAX_TURNS = int(
    os.environ.get("TMAX_CALL_LIMIT", os.environ.get("SWE_MAX_TURNS", "64"))
)
# Per-turn generation length is governed by the generator SamplingConfig.max_tokens
# (the AnthropicAdapter ignores the HTTP body's max_tokens); we still send a value
# for a well-formed request. Matches the config's per_turn_max_tokens 16384 (tmax
# prod), which leaves room for multiple turns under the 65536 model context.
_TURN_MAX_TOKENS = 16384
# Consecutive no-bash-call turns tolerated (with a format-error reminder) before
# stopping, so a misformatting policy cannot spin to the turn cap.
_MAX_FORMAT_ERRORS = int(os.environ.get("TMAX_MAX_FORMAT_ERRORS", "3"))
# open_instruct parity: production open_instruct (scripts/tmax/RL/qwen35_9b.sh) runs
# with tool_call_format_error_feedback=False, so a turn that parses to NO bash tool
# call TERMINATES the rollout on the FIRST occurrence (vllm_utils.py:1253 `if not
# tool_calls and not format_error_feedback: break`) -- no reminder, no retry. Match
# that by default. Set TMAX_FORMAT_ERROR_FEEDBACK=1 to restore the older behavior
# (inject the FORMAT_ERROR_TEMPLATE reminder and retry up to _MAX_FORMAT_ERRORS).
_FORMAT_ERROR_FEEDBACK = os.environ.get("TMAX_FORMAT_ERROR_FEEDBACK", "0") == "1"
# Per-bash-command wall-clock cap. A hung command (e.g. the model runs something
# interactive/infinite) otherwise blocks the whole rollout until the much larger
# whole-rollout guard. Cap each bash command at 120s to match the official TMax env
# (SWERLVanilluxSandboxEnv default timeout=120): a hung command is killed at 2 min
# so slow turns do not drag the episode. The verifier's own timeout
# (TMAX_EVAL_TIMEOUT_SEC) governs test.sh grading separately.
_EXEC_TIMEOUT = int(os.environ.get("TMAX_EXEC_TIMEOUT_SEC", "120"))

# Persistent-shell wrapper (ported verbatim in spirit from
# SWERLVanilluxSandboxEnv._BASH_WRAPPER): source saved env, cd to saved cwd, run the
# command, then persist env + cwd for the next call.
_BASH_WRAPPER_PATH = "/tmp/.tmax_vanillux_bash_wrapper.sh"
_BASH_CWD_PATH = "/tmp/.tmax_vanillux_cwd"
_BASH_ENV_PATH = "/tmp/.tmax_vanillux_env"
_BASH_COMMAND_PATH_PREFIX = "/dev/shm/.tmax_vanillux_command."
_BASH_DEFAULT_CWD = "/app"
_BASH_WRAPPER = f"""#!/bin/bash
if [ ! -r "$1" ]; then
  exit 1
fi
__torchtitan_vanillux_command=
IFS= read -r -d '' __torchtitan_vanillux_command < "$1" || :
rm -f "$1"
set -- "$__torchtitan_vanillux_command"
unset __torchtitan_vanillux_command
set -a
source {shlex.quote(_BASH_ENV_PATH)} 2>/dev/null || true
set +a
_cwd=
IFS= read -r _cwd 2>/dev/null < {shlex.quote(_BASH_CWD_PATH)} || :
[ -n "$_cwd" ] || _cwd={shlex.quote(_BASH_DEFAULT_CWD)}
cd "$_cwd" 2>/dev/null || cd /workspace || exit 1
eval "$1"
_exit_code=$?
export -p > {shlex.quote(_BASH_ENV_PATH)}
pwd > {shlex.quote(_BASH_CWD_PATH)}
exit $_exit_code
"""

# The single bash tool, exposed to the policy as an Anthropic tool schema (the
# adapter converts it to a renderers ToolSpec -> native qwen3 tool call). Name +
# description mirror the env's _BASH_TOOL so the SFT policy stays in-distribution.
_BASH_TOOL: list[dict] = [
    {
        "name": "bash",
        "description": (
            "Execute a bash command in a persistent shell. Working directory and "
            "environment variables are preserved between calls."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                }
            },
            "required": ["command"],
        },
    }
]


def render_instance(task: str) -> str:
    """Render the vanillux instance template (only ``{{task}}``, literal substitution)."""
    return INSTANCE_TEMPLATE.replace("{{task}}", task)


def truncate_observation(output: str) -> str:
    """mini-swe-agent head/tail truncation for long tool output."""
    if len(output) <= OBS_MAX_CHARS:
        return output
    elided = len(output) - OBS_HEAD_CHARS - OBS_TAIL_CHARS
    return (
        f"{OBS_TOO_LONG_HINT}\n\n"
        f"---- HEAD ({OBS_HEAD_CHARS} chars) ----\n"
        f"{output[:OBS_HEAD_CHARS]}\n"
        f"---- {elided} chars elided ----\n"
        f"---- TAIL ({OBS_TAIL_CHARS} chars) ----\n"
        f"{output[-OBS_TAIL_CHARS:]}"
    )


def _format_error(error: str) -> str:
    return FORMAT_ERROR_TEMPLATE.replace("{{error}}", error)


async def _prepare_runtime(sb: Sandbox) -> None:
    """Create the workspace dirs, an /app symlink, and install the persistent-shell
    wrapper -- a verbatim port of SWERLVanilluxSandboxEnv._prepare_vanillux_runtime
    (+ the reset-time mkdir of /output and /logs/verifier). The agent's initial cwd
    is ALWAYS /app (the env never uses a per-task workdir; instructions cd as needed),
    so the tmax SFT policy sees the same starting state it was trained under."""
    await sb.exec(
        "mkdir -p /workspace /output /logs/verifier /root && "
        "cd /workspace && "
        '[ -d /app ] || { _P="$(pwd)"; [ "$_P" != "/" ] && ln -sf "$_P" /app; } && '
        f"printf '%s\\n' {shlex.quote(_BASH_DEFAULT_CWD)} "
        f"> {shlex.quote(_BASH_CWD_PATH)} && "
        f": > {shlex.quote(_BASH_ENV_PATH)}",
        user="root",
        check=False,
        timeout=120,
    )
    await sb.write_file(_BASH_WRAPPER_PATH, _BASH_WRAPPER, user="root")
    await sb.exec(
        f"chmod +x {shlex.quote(_BASH_WRAPPER_PATH)}",
        user="root",
        check=False,
        timeout=60,
    )


async def _run_bash(sb: Sandbox, command: str, timeout: int) -> tuple[str, int]:
    """Run one command through the persistent-shell wrapper; return (raw_output, ec)."""
    encoded_command = base64.b64encode(command.encode()).decode("ascii")
    encoded_wrapper = base64.b64encode(_BASH_WRAPPER.encode()).decode("ascii")
    runtime_dirs = sorted(
        {
            os.path.dirname(_BASH_WRAPPER_PATH),
            os.path.dirname(_BASH_CWD_PATH),
            os.path.dirname(_BASH_ENV_PATH),
            os.path.dirname(_BASH_COMMAND_PATH_PREFIX),
        }
    )
    command_path_prefix = shlex.quote(_BASH_COMMAND_PATH_PREFIX)
    shell_command = (
        f"{shlex.join(['mkdir', '-p', *runtime_dirs])} || exit $?; "
        f"if [ ! -x {shlex.quote(_BASH_WRAPPER_PATH)} ]; then "
        f"printf %s {shlex.quote(encoded_wrapper)} | base64 -d "
        f"> {shlex.quote(_BASH_WRAPPER_PATH)} && "
        f"chmod +x {shlex.quote(_BASH_WRAPPER_PATH)} || exit $?; "
        "fi; "
        f"[ -f {shlex.quote(_BASH_CWD_PATH)} ] || "
        f"printf '%s\\n' {shlex.quote(_BASH_DEFAULT_CWD)} "
        f"> {shlex.quote(_BASH_CWD_PATH)}; "
        f"[ -f {shlex.quote(_BASH_ENV_PATH)} ] || "
        f": > {shlex.quote(_BASH_ENV_PATH)}; "
        f"_tt_vanillux_command_path={command_path_prefix}$$; "
        f"if printf %s {shlex.quote(encoded_command)} | base64 -d "
        '> "$_tt_vanillux_command_path"; then '
        f"bash {shlex.quote(_BASH_WRAPPER_PATH)} "
        '"$_tt_vanillux_command_path"; '
        "_tt_vanillux_rc=$?; "
        "else _tt_vanillux_rc=$?; fi; "
        'rm -f "$_tt_vanillux_command_path"; '
        '(exit "$_tt_vanillux_rc")'
    )
    ec, out, err = await sb.exec(
        shell_command,
        user="root",
        timeout=timeout,
        check=False,
    )
    output = out or ""
    if err:
        output += f"\n{err}" if output else err
    return output, ec


async def run_vanillux_loop(
    sb: Sandbox,
    *,
    task: str,
    session_id: str,
    adapter: "AnthropicAdapter",
    time_budget_sec: int,
    max_turns: int = _MAX_TURNS,
    exec_timeout: int = _EXEC_TIMEOUT,
) -> tuple[int, bool, int, str]:
    """Drive the faithful Vanillux bash-only ReAct agent against the adapter.

    ``task`` is the instruction text (rendered into the vanillux instance template).
    The agent starts in /app and navigates as the instruction directs (the env uses
    no per-task workdir). Returns ``(turns, submitted, total_format_errors,
    finish_reason)``; the rollouter grades only when ``submitted`` is True (tests
    run on the submit marker, matching the env). Never raises for a bad turn --
    format errors are surfaced to the agent as observations.

    Each turn calls ``adapter.complete`` directly in-process (no loopback HTTP):
    the shim does the Anthropic<->renderers translation + TITO turn capture, and
    returns the same Anthropic message dict the HTTP path would.
    """
    await _prepare_runtime(sb)

    messages: list[dict] = [{"role": "user", "content": render_instance(task)}]
    start_time = time.time()
    deadline = start_time + time_budget_sec
    turns = 0
    submitted = False
    finish_reason: str | None = None
    consecutive_format_errors = 0
    # Total format errors across the rollout (does NOT reset on a good turn, unlike
    # consecutive_format_errors) -- surfaced as a wandb metric by the rollouter.
    total_format_errors = 0

    while turns < max_turns and time.time() < deadline:
        payload = {
            "model": ADAPTER_MODEL_NAME,
            "system": SYSTEM_TEMPLATE,
            "messages": messages,
            "tools": _BASH_TOOL,
            "max_tokens": _TURN_MAX_TOKENS,
            "stream": False,
        }
        # Direct in-process call: complete() returns None only when the session is
        # closed or the generator yields nothing -> end the trajectory.
        data = await adapter.complete(session_id, payload)
        if data is None:
            finish_reason = "stopped_early"
            break

        turns += 1
        blocks = data.get("content") or []
        stop_reason = data.get("stop_reason")
        # Echo the assistant turn verbatim so the next request hash-matches and
        # the adapter TITO-appends (one packed episode).
        messages.append({"role": "assistant", "content": blocks})

        tool_uses = [
            b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"
        ]
        if not tool_uses:
            text = "".join(
                b.get("text", "")
                for b in blocks
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
            total_format_errors += 1
            consecutive_format_errors += 1
            # open_instruct parity (default): with tool_call_format_error_feedback
            # off, a turn with no bash tool call ends the rollout on the FIRST
            # occurrence -- no reminder, no retry (vllm_utils.py:1253). This is the
            # production 9B behavior, so it is our default.
            if not _FORMAT_ERROR_FEEDBACK:
                logger.info(
                    "[vanillux] %s: stopping (no tool call, format_error_feedback off)",
                    session_id,
                )
                finish_reason = "stopped_early"
                break
            # Opt-in legacy behavior (TMAX_FORMAT_ERROR_FEEDBACK=1): inject the
            # vanillux reminder and retry, bounded. At a full context the truncated
            # turn is empty and a reminder just yields another empty turn -> stop.
            if (
                stop_reason == "max_tokens" and not text
            ) or consecutive_format_errors > _MAX_FORMAT_ERRORS:
                logger.info(
                    "[vanillux] %s: stopping (empty_trunc=%s, format_errors=%d)",
                    session_id,
                    stop_reason == "max_tokens" and not text,
                    consecutive_format_errors,
                )
                finish_reason = "stopped_early"
                break
            messages.append(
                {
                    "role": "user",
                    "content": _format_error(
                        "Your last response did not include a valid `bash` tool call."
                    ),
                }
            )
            continue

        consecutive_format_errors = 0
        # The env executes EXACTLY ONE action per step (tool_calls[0]); the SFT
        # distribution is one bash call per turn. Execute only the first tool_use
        # and return an "ignored" note for any extras (keeps the tool_result <->
        # tool_use pairing valid for TITO re-rendering).
        results: list[dict] = []
        action = tool_uses[0]
        name = action.get("name", "")
        inp = action.get("input") if isinstance(action.get("input"), dict) else {}
        obs = ""
        if name != "bash":
            obs = _format_error(
                f"Unknown tool '{name}'. The only available tool is `bash`."
            )
        else:
            command = str(inp.get("command", "")).strip()
            if not command:
                obs = _format_error("'command' parameter is required.")
            else:
                raw, ec = await _run_bash(sb, command, exec_timeout)
                if SUBMIT_MARKER in raw:
                    submitted = True
                else:
                    body = truncate_observation(raw) if raw else "(no output)"
                    obs = f"{body}\n\n(exit_code={ec})"
        if submitted:
            finish_reason = "submit"
            break
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": action.get("id", ""),
                "content": obs,
            }
        )
        for extra in tool_uses[1:]:
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": extra.get("id", ""),
                    "content": _format_error(
                        "Only the first `bash` call per turn is executed; this "
                        "extra tool call was ignored."
                    ),
                }
            )
        messages.append({"role": "user", "content": results})

    # Classify how the loop ended so a run can measure the nonsubmit split
    # (time-budget wall vs turn cap vs early stop). grep the log for finish= to
    # count: submit / hit_time_budget / hit_max_turns / stopped_early.
    elapsed = time.time() - start_time
    if finish_reason is None:
        if turns >= max_turns:
            finish_reason = "hit_max_turns"
        elif time.time() >= deadline:
            finish_reason = "hit_time_budget"
        else:
            finish_reason = "stopped_early"
    logger.info(
        "[vanillux] %s: finished after %d turns (submitted=%s finish=%s elapsed=%.0fs)",
        session_id,
        turns,
        submitted,
        finish_reason,
        elapsed,
    )
    return turns, submitted, total_format_errors, finish_reason
