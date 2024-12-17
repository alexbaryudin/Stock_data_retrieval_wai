import os
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, validator
from fastapi import FastAPI, HTTPException
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_ibm import WatsonxLLM

# Load environment variables
load_dotenv()

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Credentials
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": os.getenv("WATSONX_API_KEY"),
    "project_id": os.getenv("WATSON_ML_PROJECT")
}

if not credentials["apikey"] or not credentials["project_id"]:
    logger.error("Missing WatsonX API credentials")
    raise EnvironmentError("Ensure WATSONX_API_KEY and WATSON_ML_PROJECT are set.")

# Model parameters
model_param = {
    "decoding_method": "greedy",
    "temperature": 0,
    "min_new_tokens": 5,
    "max_new_tokens": 500
}

# FastAPI app
app = FastAPI()

# Pydantic models for validation
class Query(BaseModel):
    Question: str

    @validator("Question")
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty.")
        return v

class Query_Response(BaseModel):
    result: str


def process_query(Question: str) -> str:
    try:
        logger.info(f"Processing question: {Question}")

        # Connect to the database
        db = SQLDatabase.from_uri("sqlite:///stocksDB.db")
        logger.info(f"Connected to database. Tables: {db.get_usable_table_names()}")

        # Initialize the WatsonxLLM model
        llm = WatsonxLLM(
            model_id="meta-llama/llama-3-2-90b-vision-instruct",
            url=credentials["url"],
            apikey=credentials["apikey"],
            project_id=credentials["project_id"],
            params=model_param
        )

        # Create the agent
        agent_executor = create_sql_agent(llm, db=db, verbose=True, handle_parsing_errors=True)

        # Run the query
        final_state = agent_executor.invoke(Question)
        return final_state.get("output", "No output generated.")
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        return "An error occurred while processing your request."


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/Question", response_model=Query_Response)
async def question_api(q: Query):
    try:
        result = process_query(q.Question)
        return Query_Response(result=result)
    except Exception as e:
        logger.error(f"Error processing API request: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")


def interactive_mode():
    """Run the app in interactive mode."""
    print("Welcome to the WatsonX SQL Query Processor!")
    print("Type your question below or 'exit' to quit.")

    while True:
        # Get user input
        Question = input("\nYour question: ").strip()
        if Question.lower() in {"exit", "quit"}:
            print("Exiting. Goodbye!")
            break

        # Validate and process the question
        try:
            validated_query = Query(Question=Question)
            response = process_query(validated_query.Question)
            print(f"\nResponse: {response}")
        except ValueError as e:
            print(f"Invalid input: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        # Run in interactive mode
        interactive_mode()
    else:
        # Run FastAPI with uvicorn
        import uvicorn
        uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
