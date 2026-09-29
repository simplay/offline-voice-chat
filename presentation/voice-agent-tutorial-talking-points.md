# Talking Points: Tutorial on Building a Local Voice Agent

## Slide 1: Tutorial on Building a Local Voice Agent

1. Goal: a voice assistant that runs entirely on this machine, with no cloud and offline once the models are downloaded.
2. "Existing models": nothing is trained. We combine open models: Moonshine for speech recognition and synthesis, and Qwen3.5-4B in llama.cpp for the answers.
3. The real work is the Python harness that connects them: streaming, interruption, memory, and safe tools.
4. Plan: live demo first, then architecture and program flow, then one slide per component.

## Slide 2: Live Demo [1]

## Slide 3: Voice Agent Architecture [2]

1. Read the diagram in numbered order: the core, then speech in and out, then context, then control.

2. **1 Language model:** Qwen3.5-4B, 4-bit GGUF, about 2.5 GB. Reasoning ("thinking") output is disabled so it answers directly.
3. **2 Model runtime:** llama.cpp loads the model once and streams tokens. It runs on CPU or GPU (CUDA or Metal). Measured on an RTX 3080: first text in under 0.3 s on GPU, compared with 3 to 7 s on CPU.
4. **3 Speech-to-text and 4 text-to-speech:** Moonshine, both local. Everything between them is plain text.
5. **5 System prompt and 6 memory:** context that is added to every request.
6. **7 Barge-in:** the microphone stays active, so the newest recorded speech can replace the current reply.
7. **8 Harness and tools:** the Python application that ties everything together. It handles one turn at a time, confirms that the reply was played, and has narrow privileges.
8. Takeaway: the models are replaceable parts; the behavior comes from the harness.

## Slide 4: Voice Agent Program Flow [3]

1. Initialize once: load Qwen in llama.cpp (with an optional warm-up) and the Moonshine speech models. The system prompt and saved memory are loaded too. Then the microphone opens.
2. Every turn runs the same loop, one turn at a time, on a single conversation worker.
3. **Receive transcript:** Moonshine reports a completed utterance. Only the newest waiting utterance is kept.
4. **Route request:** an exact registered phrase goes to its command. Anything else builds a prompt from the system prompt, memory, recent turns, an optional file, and the request.
5. **Produce answer:** the command handler returns one sentence, or Qwen streams tokens.
6. **Synthesize and play:** text is streamed into speech synthesis while it is still being produces.
7. **Finalize turn:** turn's output is kept in history, and (depending on the context) also in memory.
8. Recognition keeps running in parallel the whole time (makes barge-in possible).

## Slide 5: Speech Recognition and Synthesis [4]

1. The pipeline, left to right: microphone, Moonshine ASR, prompt plus Qwen, Moonshine TTS, speaker.
2. The ASR model is streaming. It produces transcripts (every 0.25 s) while you talk and a transcript when you pause.
3. Streaming output => tokens go to speech synthesis.
4. Playback stores only one audio chunk, and faster synthesis waits for it.
5. Playback writes audio in blocks of about 40 ms.

## Slide 6: System Prompt [5]

1. The system prompt is an instruction sent as the first message of every request.

## Slide 7: Short-Term and Long-Term Memory [6]

1. Short-term: the last 8 question-and-answer pairs, word for word.
2. Long-term: after each turn, simple text rules specify annotations (facts, preferences, decisions, and commitments, and the last topic).
3. The summary is stored as a JSON file in the user's application-data folder.
4. After a restart, the history is lost, but the summary will be restored.

## Slide 8: Barge-In and Cancellation [7]

1. The microphone stays on when the agent speaks, so you can interrupt by talking.
2. **Detect:** the difficulty is that the agent hears itself. If no answer has been spoken yet, it cancels immediately. Otherwise it compares the first transcript with what it is saying and ignores a match. On Linux, WebRTC acoustic echo cancellation also removes the speaker signal from the microphone.
3. **Cancel:** set the cancellation flags. llama.cpp stops at its next step through an abort callback (within ms) and playback stops after the current 40 ms block.
4. **Clean up:** a separate worker stops speech synthesis, so recognition is never blocked.
5. **Continue:** only the latest completed speech. Counters prevent old requests from starting.

## Slide 9: Harness and Tools [8]

1. harness = a Python application that connects the components, processes one turn at a time.
2. Routing: the transcript is normalized (case and punctuation). On a match, the handler's answer is spoken directly without invkoking a callback.
3. Commands: time, date, help, and reset.
