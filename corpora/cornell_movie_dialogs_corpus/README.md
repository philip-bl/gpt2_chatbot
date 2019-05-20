# Cornell Movie Dialogs Corpus

Dataset source: <http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>

Some dialogs, as they are presented in the original dataset, are broken into parts even though they should be. Hence I have processed them a little bit: for each 3-tuple (movie_id, character1_id, character2_id) I sorted all records belonging to this 3-tuple in the order in which they appear in *movie_conversations.txt*. Then, for every two records which are next to each other, if their line numbers form a consecutive range of integers, I join them together, e.g., `['L401', 'L402', 'L403']` and `['L404', 'L405', 'L406', 'L407']` are joined together. The rationale behind this is the fact that sometimes in the scripts dialogs look like:

> Cameron: Blah blaaah.
> Alice: Oh yeah
>
> Cameron looks at the dog.
>
> Cameron: Weather's good ain't it?
> Alice: Pretty fucking good.

In such cases makers of the dataset broke the dialog into two parts even though it's actually one dialog. Texts grouped using this rule can be found in a json with a list of lists at [grouped_if_same_and_consecutive.json](grouped_if_same_and_consecutive.json). This file was produced by [extract.py](extract.py).


