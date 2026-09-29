# LESSONS

Rules for myself, written after mistakes made building this pipeline.

## Say "inference, not training" before the question is asked
When running a pre-trained model, state plainly that nothing is being trained,
and give the cost in concrete terms (N frames x one forward pass, minutes on
CPU). Left unsaid, "running a model" reads as training, and the conversation
detours into hardware that would not have helped.

## Never install into a shared environment
`pip install` into `pygame_env` silently upgraded cv2 4.11 -> 5.0.0 for every
other project using it. Create a project venv *first*, before the first install,
not after noticing the damage.

## Check the interpreter version before creating a venv
On macOS `/opt/homebrew/bin/python3` was 3.14, which had no wheels for the
dependency and a broken ensurepip. Run `python3 -V` and name the version
explicitly (`python3.12 -m venv`) rather than trusting the bare `python3` alias.

## A zero result from a detector is a claim that needs a positive control
`ffmpeg`'s scene filter reported 0 cuts twice -- but the output was empty
because `-v error` suppressed the very lines being parsed. Zero findings and a
broken harness look identical. Confirm the detector emits *something* on input
known to contain the thing, before reporting an absence.

## zsh does not word-split unquoted variables
`set -- $pair` inside a loop passes the whole string as one argument, unlike
bash. Use a shell function with explicit arguments, or an array.

## Prefer the newest library version only when it is the tested one
mediapipe 1.0.1 crashed on macOS ARM inside DrishtiMetalHelper at graph
construction, ignoring the CPU delegate. Pinning 0.10.21 fixed it outright.
When a hard crash sits inside vendor GPU code, try the older well-trodden
release before debugging their graph.

## State a prediction before running the experiment
Predicting that a classical tutu would beat a romantic one *before* processing
Paquita made the result a test rather than a rationalisation. Cheap to do, and
it keeps the analysis honest.
