import re
import os

def process_dialogue(text):
    """
    Process a dialogue text with emotion tags into a structured dictionary.
    
    Args:
    text (str): Input text containing dialogue with emotion tags
    
    Returns:
    dict: Structured dialogue data with speakers, tones, and sentences
    """
    dialogue_data = {}
    index = 1
    
    # Split the text into lines and remove empty lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    for line in lines:
        # Skip lines that don't contain dialogue
        if ':' not in line or not any(char.isalpha() for char in line):
            continue
            
        # Extract components
        try:
            # Extract initial tone(s)
            initial_tones = []
            if line.startswith('['):
                tone_end = line.find(']')
                if tone_end != -1:
                    tones = line[1:tone_end].strip()
                    initial_tones = [tone.strip() for tone in tones.split(',')]
                    line = line[tone_end + 1:].strip()
            
            # Split speaker and dialogue
            speaker_part, dialogue_part = line.split(':', 1)
            speaker = speaker_part.strip()
            dialogue = dialogue_part.strip()
            
            # Extract end tone(s)
            end_tones = []
            if dialogue.endswith(']'):
                tone_start = dialogue.rfind('[')
                if tone_start != -1:
                    tones = dialogue[tone_start + 1:-1].strip()
                    end_tones = [tone.strip() for tone in tones.split(',')]
                    dialogue = dialogue[:tone_start].strip()
            
            # Create dictionary entry
            dialogue_data[index] = {
                'speaker': speaker,
                'tone': initial_tones,
                'sentence': dialogue,
                'end_tone': end_tones if end_tones else initial_tones.copy()  # Use initial tones if no end tones specified
            }
            
            index += 1
            
        except Exception as e:
            print(f"Error processing line: {line}")
            print(f"Error details: {str(e)}")
            continue
    
    return dialogue_data

def change_speaker_names(dialogue_dict, name_mapping):
    """
    Change speaker names in the dialogue dictionary based on a custom mapping.
    
    Args:
    dialogue_dict (dict): Original dialogue dictionary
    name_mapping (dict): Dictionary mapping original names to new names
    
    Returns:
    dict: Updated dialogue dictionary with new speaker names
    """
    updated_dialogue = dialogue_dict.copy()  # Create a copy to avoid modifying the original
    
    for index, entry in updated_dialogue.items():
        original_speaker = entry['speaker']
        if original_speaker in name_mapping:
            entry['speaker'] = name_mapping[original_speaker]
    
    return updated_dialogue

def print_dialogue_dict(dialogue_dict):
    """
    Print the dialogue dictionary in a readable format.
    """
    for index, entry in dialogue_dict.items():
        print(f"\nEntry {index}:")
        print(f"  Speaker: {entry['speaker']}")
        print(f"  Tone: {entry['tone']}")
        print(f"  Sentence: {entry['sentence']}")

def remove_last_bracket_tag(line):
    # Regular expression to find the last [tag]
    return re.sub(r'\[.*?\]$', '', line).strip()

def process_file(file_path, skip_lines=0):
    result = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i < skip_lines:
                continue
            if line.strip():
                result.append(line.strip() + '\n')
    return " ".join(result)

def get_highest_index(folder_path):
    max_index = -1  
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            try:
                # Extract the index by removing the .wav extension and converting to integer
                index = int(file_name.replace(".wav", ""))
                max_index = max(max_index, index)
            except ValueError:
                # Ignore files that don't have a valid integer index
                continue
    return max_index

if __name__ == "__main__":
    file_content = process_file("podcast_script.txt", skip_lines=6)
    custom_mapping = {
        "Alex": "Emma",
        "Chris": "James"
    }
    file_dict = process_dialogue(file_content)
    
    custom_result = change_speaker_names(file_dict, custom_mapping)
    print("\nWith custom speaker names:")

    from parl_gen import parl_loader, describe_speaker, audio_generator 
    model, tokenizer, description_tokenizer = parl_loader()
    speaker_tokens = {}
    print(custom_result)
    max_idx = get_highest_index("audio")
    for index, entry in custom_result.items():
        print(f"\nEntry {index}:")
        print(f"  Speaker: {entry['speaker']}")
        print(f"  Tone: {entry['tone']}")
        print(f"  Sentence: {entry['sentence']}")
        if max_idx >= index:
            continue 
        if entry['speaker'] not in speaker_tokens.keys():
            speaker_tokens[entry['speaker']] = list(describe_speaker(description_tokenizer, speaker=entry['speaker']))
        print(len(speaker_tokens))
        audio_generator(model, tokenizer, speaker_tokens[entry['speaker']][0], speaker_tokens[entry['speaker']][1],"cpu", entry['sentence'], str(index), entry['speaker'], "happy")
        print("-" * 40)  # Separator for readability