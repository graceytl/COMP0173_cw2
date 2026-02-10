"""
Script to generate statements using vLLM API and save outputs for further processing"""

from src.gen_dataset.prompts import STATEMENT_GEN
from src.gen_dataset.models import GeneratedStatements, StatementMetadata
from openai import OpenAI
import logging
from src.gen_dataset.themes import tasks
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Connect to local vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=""
)

def generate_statements(task, model="FuseAI/FuseChat-7B-VaRM"):
    """
    Generate statements using OpenAI SDK with vLLM API
    """
    prompt = STATEMENT_GEN.format(
        theme=task['theme'], 
        flags=", ".join(task['flags']), 
        examples="\n".join(task.get('examples', []))
    )
    print(f"Generating statements related to {task['theme']}")

    json_schema = GeneratedStatements.model_json_schema()

    statements = []

    try:
        print("Sending request to FuseChat via vLLM...")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=None,
            top_p=0.75,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "statement_list",
                    "schema": json_schema
                }
            },
            stop=None,
        )

        logging.info("Response received from model.")
        logging.info("Parsing.")
    
        if response.choices and len(response.choices) > 0:
            text = response.choices[0].message.content

            try:
                statements = GeneratedStatements.model_validate_json(text)
            except Exception as e:
                logging.error(f"Error parsing JSON response: {e}. Returning raw text.")
                return text
        else:
            logging.info("No response received. Returning response object.")
            return response
    except Exception as e:
        logging.error(f"Error during statement generation: {e}")

    return statements

def save_response(responses, path):
    """
    Save output to a pickle file
    """
    with open(path, "wb") as f:
        pickle.dump(responses, f)

if __name__ == "__main__":
    logging.info("Starting task generation...")

    malformed = []
    output_path = Path.cwd() / "src/gen_dataset/outputs/"
    output_path.mkdir(parents=True, exist_ok=True)

    tasks = [tasks[25]]

    for i, task in enumerate(tasks):
        logging.info(f"\n{'='*50}\nTask {i+1}/{len(tasks)}: '{task['theme']}'\n{'='*50}")

        response = generate_statements(task)

        if type(response) == GeneratedStatements:
            logging.info(f"{task['theme']}: {len(response.statements)} statements generated.")

            logging.info(f"Processing {len(response.statements)} statements.")
            structured = []
            for statement in response.statements:
                try:
                    statement_metadata = StatementMetadata(
                        statement = statement,
                        cleaned_text = statement.marked_text.replace("===", ""),
                        themes = [task['category']],
                        target_groups = [task['target_group']],
                        data_source = "manual_curation",
                        is_harmful = True if statement.type == "Normative" else False  # NOTE: Placeholder - requires manual labelling
                    )
                    structured.append(statement_metadata)
                except Exception as e:
                    logging.error(f"Error processing statement: {e}. Statement: {statement}")
                    save_response(statement, output_path / f"malformed_statement_{task['target_group']}_2.pkl")
                    logging.info(f"Malformed statement saved to outputs/malformed_statement_{task['target_group']}_2.pkl for further processing.")
            logging.info(f"{len(structured)} responses parsed correctly")
            save_response(structured, output_path / f"statements_{task['target_group']}_2.pkl")
            logging.info(f"Structured response saved to outputs/statements_{task['target_group']}_2.pkl for further processing.")
        else:
            logging.warning(f"{task['theme']}: Malformed response received.")
            save_response(response, output_path / f"malformed_response_{task['target_group']}_2.pkl")
            logging.info(f"Malformed response saved to outputs/malformed_response_{task['target_group']}_2.pkl for further processing.")