from orchestrator.crew_manager import create_enhanced_crew_manager

# Initialize the shared crew manager
crew_manager = create_enhanced_crew_manager()

def handle_morning_brief(query: str) -> dict:
    """
    Handles the morning brief use case.
    
    Args:
        query (str): User's query string (typically from voice or UI input).
    
    Returns:
        dict: Response text or error.
    """
    try:
        result = crew_manager.process_voice_query(query)
        return {"response": result}
    except Exception as e:
        return {"error": f"Workflow failed: {str(e)}"}
