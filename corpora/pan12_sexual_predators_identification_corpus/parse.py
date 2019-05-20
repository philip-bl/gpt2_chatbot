import xml.etree.ElementTree as ElementTree
from xml.sax.saxutils import unescape
from itertools import chain

from typing import *

from libcrap import save_json


HTML_ESCAPE_SEQUENCES = {
    "&quot;": '"',
    "&apos;": "'",
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">"
}

def get_authors_ids(conversation: ElementTree.Element) -> Tuple[str, str]:
    """Given a node of type "conversation", returns a tuple of its 2 authors ids."""
    assert conversation.tag == "conversation"
    result = tuple(frozenset(node.text for node in conversation.iter("author")))
    return result


def parse_conversation(conversation: ElementTree.Element):
    assert conversation.tag == "conversation"
    authors = get_authors_ids(conversation)
    if len(authors) != 2:
        return None
    if conversation[0].find("author").text != authors[0]:
        authors = authors[::-1]
    for message in conversation:
        author = authors.index(message.find("author").text) # 0 or 1
        text = message.find("text").text
        if text is not None:
            # There are escape html sequences such as &apos;
            # let's unescape them
            text_unescaped = unescape(text, HTML_ESCAPE_SEQUENCES)
            yield {"author": author, "text": text_unescaped}


def load_conversations_from_xml(file_path: str) \
-> Tuple[Tuple[Dict[str, Union[int, str]], ...], ...]:
    xml_tree = ElementTree.parse(file_path)
    root = xml_tree.getroot()
    assert all(node.tag == "conversation" for node in root)
    conversations = (
        parse_conversation(conversation) for conversation in root
    )
    conversations_without_nones = filter(lambda x: x is not None, conversations)
    tupled_conversations = (
        tuple(conversation) for conversation in conversations if conversation
    )
    return tuple(conv for conv in tupled_conversations if conv)

conversations_train = load_conversations_from_xml(
    "/home/shibbiry/archive/datasets/pan12_sexual_predator_identification_corpus/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
)
conversations_test = load_conversations_from_xml(
    "/home/shibbiry/archive/datasets/pan12_sexual_predator_identification_corpus/pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml"
)
save_json(
    tuple(chain(conversations_train, conversations_test)),
    "conversations_train_and_test.json"
)
all_symbols = tuple(chain.from_iterable(x["text"] for x in chain.from_iterable(chain(conversations_train, conversations_test))))
print(f"Total number of symbols = {len(all_symbols)}")

