import re
import argparse

# def get_token_positions(text, classes):
#     """
#     Finds the start and end positions of all occurrences of class words or phrases in the text,
#     and concatenates all the spans into a single 2-level nested list.

#     Args:
#         text (str): The input text to search.
#         classes (list): A list of class words or phrases to find.

#     Returns:
#         list: A concatenated 2-level nested list containing spans for all classes.
#     """
#     all_spans = []  # This will hold the concatenated spans for all classes

#     for cls in classes:
#         # Use regex to find all matches of the class in the text
#         for match in re.finditer(re.escape(cls), text):
#             words = cls.split()  # Handle multi-word classes
#             span = []
#             start = match.start()

#             for word in words:
#                 word_start = text.find(word, start)
#                 word_end = word_start + len(word)
#                 span.append([word_start, word_end])
#                 start = word_end  # Update start for next word in multi-word class

#             all_spans.append(span)  # Append the spans directly (no extra list nesting)
    
#     return all_spans

import re

def get_token_positions(text, classes):
    """
    Finds the start and end positions of all occurrences of class words or phrases in the text,
    ensuring that overlapping matches are not reused for different classes.
    Longer classes are prioritized over shorter ones.

    Args:
        text (str): The input text to search.
        classes (list): A list of class words or phrases to find.

    Returns:
        list: A concatenated 2-level nested list containing spans for all classes.
    """
    all_spans = []  # This will hold the concatenated spans for all classes
    used_positions = set()  # To keep track of positions already matched

    # Sort classes by length (longest first) to prioritize longer matches
    classes = sorted(classes, key=len, reverse=True)

    for cls in classes:
        # Use regex to find all matches of the class in the text
        for match in re.finditer(re.escape(cls), text):
            words = cls.split()  # Handle multi-word classes
            span = []
            start = match.start()

            valid_match = True  # Flag to check if this match overlaps with used positions
            for word in words:
                word_start = text.find(word, start)
                word_end = word_start + len(word)

                # Check if any position in this word range has already been used
                if any(pos in used_positions for pos in range(word_start, word_end)):
                    valid_match = False
                    break

                span.append([word_start, word_end])
                start = word_end  # Update start for next word in multi-word class

            if valid_match:
                all_spans.append(span)
                # Mark these positions as used
                for word_start, word_end in span:
                    used_positions.update(range(word_start, word_end))
    
    return all_spans


if __name__ == "__main__":

    # # Input Text Prompt
    # text_prompt = (
    #     "an excavator is digging a trench. a pipe is lying on dirt. a pipe is covered in dirt. "
    #     "a shovel is lying on dirt. a shovel is covered in dirt. a tool is lying on dirt. "
    #     "a tool is covered in dirt. a single large stone is lying on dirt. a single large stone is covered in dirt. "
    #     "a construction barrier is lying on dirt. a construction barrier is covered in dirt. "
    #     "a tube is lying on dirt. a tube is covered in dirt. a cable is lying on dirt. a cable is covered in dirt."
    # )

    # # Classes
    # classes = ["excavator", "pipe", "shovel", "tool", "large stone", "barrier", "tube", "cable"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--text-prompt", default=None)
    parser.add_argument("--classes", default=None)
    args = parser.parse_args()

    # Parse the classes string into a list
    classes = args.classes.split(",") if args.classes else []

    # Get token positions
    positions = get_token_positions(args.text_prompt, classes)

    # Print results
    print("--text-prompt ", "\"" + args.text_prompt + "\"", "--token-spans ", "\"",positions,"\"")
