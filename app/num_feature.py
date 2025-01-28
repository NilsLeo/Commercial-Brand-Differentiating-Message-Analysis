import spacy

# Lade das englische SpaCy-Modell
nlp = spacy.load("en_core_web_sm")

def find_first_modifier_with_degree(token, visited=None, first_modifier=None):
    if visited is None:
        visited = set()
    visited.add(token)

    # Blacklist für unerwünschte Modifier
    blacklist = {"old"}  # Füge hier weitere unerwünschte Wörter hinzu

    # Check if the next token is % directly after the number
    token_text = token.text
    has_percent = False
    try:
        next_token = token.nbor(1)  # Get the next token
        if next_token.text == "%":  # If the next token is %, combine it with the number
            token_text = "%"
            has_percent = True
    except IndexError:
        pass  # If there is no next token, skip this step

    # Check if the current token is an adjective or adverb with a defined Degree
    if token.pos_ in {"ADJ", "ADV"}:
        if token.text in blacklist:  # Skip blacklisted words
            return None
        degree = token.morph.get("Degree")
        if degree:  # Only return if Degree is defined (e.g., Cmp or Sup)
            modifier = f"{token.text}"
            if first_modifier is None or token.text != first_modifier:  # Ensure no repetition
                return modifier

    # Check children for modifiers
    for child in token.children:
        if child not in visited:
            modifier = find_first_modifier_with_degree(child, visited, first_modifier)
            if modifier:
                # Skip if the modifier is blacklisted
                if any(blacklisted_word in modifier for blacklisted_word in blacklist):
                    continue
                # Combine % with the modifier if found
                return f"{token_text} {modifier}" if has_percent else modifier

    # Check the head for modifiers
    if token.head not in visited and token.head != token:
        modifier = find_first_modifier_with_degree(token.head, visited, first_modifier)
        if modifier:
            # Skip if the modifier is blacklisted
            if any(blacklisted_word in modifier for blacklisted_word in blacklist):
                return None
            # Combine % with the modifier if found
            return f"{token_text} {modifier}" if has_percent else modifier

    # Return the token text with % if no modifier found
    return token_text if has_percent else None


def extract_features(text):
    doc = nlp(text)
    features = []
    for token in doc:
        if token.like_num:  # Check if the token is a number
            #print(f"Found number: {token.text}")
            # Find the first modifier related to the number
            modifier = find_first_modifier_with_degree(token)
            if modifier and modifier.lower() != token.text.lower():
                # Search for the modifier in the document
                if not any(tok.text == modifier and tok.like_num for tok in doc):
                    features.append((token.text, modifier))
    return features, len(features)

def apply_on_transcript(text):
    found_pairs = extract_features(text)
    #print(f"Founded {(found_pairs[1])} pairs")
    return found_pairs

#test---------------------------
#text = """Our product is 10 times faster and 50% more efficient. """
#print(apply_on_transcript(text))