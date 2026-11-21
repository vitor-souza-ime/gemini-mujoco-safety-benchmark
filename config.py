"""V13: 50 new live Gemini scenes with crash-safe per-scene checkpoints."""

# Google AI Studio API key. Keep it private.
GEMINI_API_KEY = "SUA API KEY"

GEMINI_MODEL = "gemini-robotics-er-2-preview"

# Scientific batch: scene IDs 21..70 = 50 new scenes.
# Combined with the already completed V11 scenes 1..20 -> 70 unique scenes total.
START_SCENE = 21
TARGET_SCENES = 50
MAX_LIVE_API_CALLS_PER_INVOCATION = 50

# Structured JSON output.
GEMINI_MAX_OUTPUT_TOKENS = 4096

# Conservative pacing. The timestamp of the previous attempt is persisted,
# so restarting the program does not immediately violate this pacing.
API_RPM_LIMIT = 1
API_RATE_SAFETY_MARGIN_SECONDS = 1.5

# No automatic HTTP retry. On 429/network/timeout the program stops safely.
# Run 05_resume... after the quota/connection is available again.
REQUEST_CONNECT_TIMEOUT_SECONDS = 10
REQUEST_READ_TIMEOUT_SECONDS = 90

SEED_BASE = 20261121
CAPTURE_ALL_INPUT_IMAGES = True
