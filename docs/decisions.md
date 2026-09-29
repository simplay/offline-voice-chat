# Design Decisions

Why the code works the way it does.

- The voice agents performs an online search when a users says "please check online".
- Speech starts before the model has finished, so answers feel faster.
- Moonshine splits sentences at short pauses, so the agent waits about a second of silence before answering (can be specified using `--end-of-turn-delay`).
- The agent can be interrupted any time. Stopping an answer happens on another thread, to ensure that listening never pauses.
- Interrupted answers are dropped from history, because we can't tell how much of it you heard.
- The model's own voice is ignored.
