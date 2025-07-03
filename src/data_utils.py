"""Prompt generation and handling functions."""

import itertools
from typing import List, Dict, Tuple


def generate_prompts(
    prompt_templates: List[str],
    mediums: List[str],
    contents: List[str],
    styles: List[str],
) -> List[Dict[str, str]]:
    """
    Generates all unique prompt combinations.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
            - "template": The prompt template string.
            - "text": The generated prompt string.
            - "content_word": The full content string.
            - "style_word": The full style string.
            - "medium_word": The full medium string.
    """
    all_prompts_info = []
    prompt_combinations = list(
        itertools.product(*[prompt_templates, mediums, contents, styles])
    )

    for template, medium, content, style in prompt_combinations:
        prompt_text = template.replace("<MEDIUM>", medium)
        prompt_text = prompt_text.replace("<CONTENT>", content)
        prompt_text = prompt_text.replace("<STYLE>", style)
        prompt_info = {
            "text": prompt_text,
            "template": template,
            "medium_word": medium,
            "content_word": content,
            "style_word": style,
        }
        all_prompts_info.append(prompt_info)

    return all_prompts_info


def get_token_indices_for_template_words(
    tokenizer,
    prompt_text: str,
    template_str: str,
    content_word: str,
    style_word: str,
    medium_word: str,
) -> Tuple[List[str], List[int], int, int, int]:
    """
    Identifies words from the template and their corresponding token indices in the tokenized prompt.

    Returns:
        A tuple containing:
        - words (List[str]): The list of identified words from the template.
        - token_indices (List[int]): The starting token index for each word.
        - content_idx (int): The index of the content_word in the `words` list.
        - style_idx (int): The index of the style_word in the `words` list.
        - medium_idx (int): The index of the medium_word in the `words` list.
    """
    template_parts = template_str.split()
    resolved_words = []
    for part in template_parts:
        if part == "<CONTENT>":
            resolved_words.append(content_word)
        elif part == "<STYLE>":
            resolved_words.append(style_word)
        elif part == "<MEDIUM>":
            resolved_words.append(medium_word)
        else:
            resolved_words.append(part)

    tokenized_prompt = [w.replace("</w>", "") for w in tokenizer.tokenize(prompt_text)]

    current_token_idx = 0
    word_start_indices = []

    for word_to_match in resolved_words:
        normalized_word_to_match = word_to_match.replace(" ", "").lower()
        for n_tokens_to_group in range(
            1, len(tokenized_prompt) - current_token_idx + 1
        ):
            current_segment_tokens = tokenized_prompt[
                current_token_idx : current_token_idx + n_tokens_to_group
            ]
            segment_str = "".join(current_segment_tokens).lower()

            if segment_str == normalized_word_to_match:
                word_start_indices.append(current_token_idx)
                current_token_idx += n_tokens_to_group
                break

    content_idx = resolved_words.index(content_word)
    style_idx = resolved_words.index(style_word)
    medium_idx = (
        resolved_words.index(medium_word) if medium_word in resolved_words else -1
    )

    return resolved_words, word_start_indices, content_idx, style_idx, medium_idx
