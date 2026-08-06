# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Vanillux harness prompts, vendored VERBATIM from mini-swe-agent v2.2.x via the
AI2 tmax repo (open_instruct/environments/vanillux_prompts.yaml). Inlined as Python
string constants (not a data file) so they always ship in the installed wheel.

These strings are the exact instruction surface the tmax Qwen3.5-9B was SFT'd/RL'd
under (system_template is byte-identical to the train-time
--system_prompt_override_file), so they are kept byte-for-byte -- including the
Unicode em dashes in SYSTEM_TEMPLATE -- to keep the policy in-distribution. Do NOT
"fix" the punctuation.
"""

# flake8: noqa: E501

SYSTEM_TEMPLATE = "You are a helpful assistant that can interact with a computer.\n\nYour response must include a THOUGHT section before your action where you\nexplain your reasoning. After the THOUGHT, you must call the `bash` tool\nwith EXACTLY ONE bash command (multiple commands chained with `&&` or `||`\ncount as a single action).\n\nFailure to follow these rules — calling no tool, calling a tool other than\n`bash`, or omitting the THOUGHT — will cause your response to be rejected.\n"

INSTANCE_TEMPLATE = "Please solve this task:\n\n{{task}}\n\nYou can execute bash commands and edit files (with `sed`, `cat > file << 'EOF'`,\netc.) to implement the necessary changes.\n\n## Recommended Workflow\n\nThis workflow should be done step-by-step so that you can iterate on your\nchanges and any possible problems.\n\n1. Analyze the codebase / environment by finding and reading relevant files.\n2. If applicable, create a script to reproduce the issue or expected behaviour.\n3. Implement the change(s) by editing the source code or environment state.\n4. Verify your fix works by running your script (or relevant test) again.\n5. Test edge cases to ensure your fix is robust.\n6. Submit your changes and finish your work by issuing the following command:\n   `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\n   Do not combine it with any other command. After this command, you cannot\n   continue working on this task.\n\n## Important Rules\n\n1. Every response must contain exactly one tool call to `bash`.\n2. Directory and environment-variable changes ARE persistent across calls\n   in this harness — you can `cd` and `export` and subsequent commands will\n   see the change. (This differs from upstream mini-swe-agent's subshell\n   model; treat the shell as a long-running login shell.)\n3. Long-running commands: wrap with `timeout`, e.g. `timeout 30 <command>`.\n4. Interactive commands are not possible. Use `yes`/`no` piping or\n   non-interactive flags as appropriate.\n5. Output may be truncated. Use `head`, `tail`, `grep`, `sed -n 'A,Bp'`,\n   etc. to filter large outputs.\n\n## Useful command examples\n\n### Create a new file:\n`cat <<'EOF' > newfile.py\nimport numpy as np\nhello = \"world\"\nprint(hello)\nEOF`\n\n### Edit files with sed:\n`sed -i 's/old_string/new_string/g' filename.py`        # all occurrences\n`sed -i '1s/old_string/new_string/' filename.py`         # first on line 1\n`sed -i '1,10s/old_string/new_string/g' filename.py`     # lines 1-10\n\n### View file content:\n`nl -ba filename.py | sed -n '10,20p'`\n"

FORMAT_ERROR_TEMPLATE = "Format error: {{error}}\n\nPlease always provide EXACTLY ONE call to the `bash` tool. If you want to\nend the task, please issue the command `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\nvia the `bash` tool, with no other content in the command.\n"

OBS_MAX_CHARS = 10000
OBS_HEAD_CHARS = 5000
OBS_TAIL_CHARS = 5000
OBS_TOO_LONG_HINT = "The output of your last command was too long.\nPlease try a different command that produces less output.\nIf you're looking at a file you can try use head, tail or sed to view a\nsmaller number of lines selectively. If you're using grep or find and it\nproduced too much output, you can use a more selective search pattern.\nIf you really need to see something from the full command's output, you\ncan redirect output to a file and then search in that file.\n"
