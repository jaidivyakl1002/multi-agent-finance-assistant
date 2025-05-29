import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from orchestrator.task_router import TaskRouter

def test_text_query():
    print("----- Testing Text Query -----")
    router = TaskRouter()
    request = {
        "input_type": "text",
        "query": "What's our risk exposure in Asia tech stocks today, and highlight any earnings surprises?"
    }
    response = router.route(request)
    print("Response:", response)


def test_voice_query_stub():
    print("----- Testing Voice Query (Stub) -----")
    router = TaskRouter()

    # Stub audio input (replace with actual binary or file path if implemented)
    fake_audio = "audio/path/or/bytes"  # Placeholder; adjust as per your VoiceAgent expectations

    request = {
        "input_type": "voice",
        "audio": fake_audio
    }
    response = router.route(request)
    print("Response:", response)


if __name__ == "__main__":
    test_text_query()
    # test_voice_query_stub()  # Uncomment if STT is implemented
