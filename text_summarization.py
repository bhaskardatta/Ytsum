import requests
import json
import re
import sys

# Hugging Face API URL and token
API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-32B-Instruct"
API_TOKEN = "your_huggingface_api_token"

headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

def query(payload):
    """
    Queries the Hugging Face Inference API with the given payload.
    """
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"API request failed: {response.text}")
    return response.json()

def generate_podcast_script(input_text):
    """
    Generates a podcast-style script with emotional annotations.
    """
    prompt = (
        "Transform the following text into a podcast-style dialogue between two people. "
        "Add an emotional tag at the beginning of each sentence to reflect the speaker's tone. "
        "Here is the input:\n\n"
        f"{input_text}\n\n"
        "Output a podcast dialogue with emotional annotations."
    )
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 1000,
            "temperature": 0.7,
        }
    }
    
    response = query(payload)
    return response[0]["generated_text"]

def extract_podcast_content(generated_text):
    """
    Extracts the podcast content from the generated text, removing the prompt or other unwanted sections.
    """
    # Assuming the content starts after "---" or the first dialogue appears
    content_start = generated_text.find("---")
    if content_start != -1:
        return generated_text[content_start + 3:].strip()  # Remove "---" and leading spaces
    return generated_text  # Fallback to entire text if no separator is found

def main():
    # Configure the terminal to use UTF-8 encoding
    sys.stdout.reconfigure(encoding='utf-8')
    
    # Read input text from a file
    input_file = "input_text.txt"
    try:
        with open(input_file, "r", encoding="utf-8") as file:
            input_text = file.read()
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return

    # Generate podcast-style dialogue
    try:
        generated_text = generate_podcast_script(input_text)
        podcast_content = extract_podcast_content(generated_text)

        output_file = "podcast_script.txt"
        
        # Save the output to a file with UTF-8 encoding
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(podcast_content)
        
        print(f"Podcast script generated successfully and saved to {output_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
