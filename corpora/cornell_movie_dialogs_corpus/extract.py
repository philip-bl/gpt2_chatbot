import argparse
import codecs
import csv
import os
from itertools import chain

from typing import Tuple

import pandas as pd

from more_itertools import consecutive_groups

from libcrap import save_json # pip install libcrap

"""
Load the cornell movie dialog corpus.

Available from here:
http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

"""

def loadLines(fileName, fields):
    """
    Args:
        fileName (str): file to load
        field (set<str>): fields to extract
    Return:
        dict<dict<str>>: the extracted fields for each line
    """
    lines = {}

    with open(fileName, 'r', encoding='iso-8859-1') as f:  # TODO: Solve Iso encoding pb !
        for line in f:
            values = line.split(" +++$+++ ")

            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]

            lines[lineObj['lineID']] = lineObj

    return lines

def loadConversations(fileName, fields):
    """
    Args:
        fileName (str): file to load
        field (set<str>): fields to extract
    Return:
        dict<dict<str>>: the extracted fields for each line
    """
    conversations = []

    with open(fileName, 'r', encoding='iso-8859-1') as f:  # TODO: Solve Iso encoding pb !
        for line in f:
            values = line.split(" +++$+++ ")

            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]

            convObj["lineIds"] = eval(convObj["utteranceIDs"])
            del convObj["utteranceIDs"]

            conversations.append(convObj)

    return conversations

def extractSentencePairs(conversations):
    """
    Extract the sample lines from the conversations
    """

    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation

        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()

            if inputLine and targetLine:  # Filter wrong samples (if one of the list is empty)
                qa_pairs.append([inputLine, targetLine])

    return qa_pairs


if __name__ == '__main__':
    dataset_dir = "dataset_source"

    lines = {}
    conversations = []

    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    lines = pd.DataFrame(
        loadLines(os.path.join(dataset_dir, "movie_lines.txt"), MOVIE_LINES_FIELDS)
            .values()
    ).set_index("lineID") # columns: character, characterID, movieID, text

    conversations = pd.DataFrame(loadConversations(
        os.path.join(dataset_dir, "movie_conversations.txt"),
        MOVIE_CONVERSATIONS_FIELDS
    )) # columns: character1ID, character2ID, lines, movieID, utteranceIDs

    # 1. separate conversations into groups (if consecutive conversations have
    # the same movie and characters)
    # 2. for each group separate it into subgroups based on consecutiveness of
    # line numbers
    def group_by_characters(conversations: pd.DataFrame) -> pd.DataFrame:
        """For each 3-tuple (movieID, character1ID, character2ID) concatenates all
        their lists of line numbers. Returns pd.Series with the 3-tuples as index
        and long tuples as values."""
        grouped = conversations.groupby(["movieID", "character1ID", "character2ID"], sort=False,)
        def groupby_apply_func(group: pd.DataFrame) -> Tuple[str]:
            return tuple(chain.from_iterable(group["lineIds"]))
        result = grouped.apply(groupby_apply_func)
        return result
    
    grouped_conversations = group_by_characters(conversations)

    ordering = lambda s: int(s[1:])
    groups_of_line_numbers = tuple(chain.from_iterable(
        tuple(tuple(group) for group in consecutive_groups(lines, ordering))        
        for lines in grouped_conversations
    )) # type: Tuple[Tuple[str, ...], ...]
    
    assert len(groups_of_line_numbers) == 60699
    
    texts = tuple(
        tuple(lines.loc[line_id, "text"] for line_id in group)
        for group in groups_of_line_numbers
    )
    movie_ids = tuple(
        lines.loc[group[0], "movieID"]
        for group in groups_of_line_numbers
    )
    assert len(texts) == len(movie_ids)
    save_json(texts, "movies_dialogs_304713_symbols.txt")
