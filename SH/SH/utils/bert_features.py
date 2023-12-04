# Generate BERT features.

class InputFeatures(object):
    """
    Inputs of the BERT model.
    """
    def __init__(self, tokens, input_ids, input_mask, input_type_ids):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, tokenizer):
    """
    Convert textual segments into word IDs.

    params
        examples: the raw textual segments in a list.
        tokenizer: a BERT Tokenizer object.

    return
        features: BERT features in a list.
    """
    import re
    features = []
    for (ex_index, example) in enumerate(examples):
        # print('example 이 무엇일까요?', example)
        # tokens = tokenizer.tokenize(example)
        # print('토큰은 이렇게 됩니다', tokens)
        # print('토큰의 길이', len(tokens))
        tokens = list()
        for ex in example:
            tokens += [letter for letter in re.sub('\s', '', ex)]
        # ['[CLS]', 음절 단위 + 띄어쓰기 제외 , '[SEP]'] => 예시에서 전체 289개가 있으면 됨

        new_tokens = []
        input_type_ids = []

        new_tokens.append("[CLS]")
        input_type_ids.append(0)
        new_tokens += tokens
        input_type_ids += [0] * len(tokens)
        new_tokens.append("[SEP]")
        input_type_ids.append(0)
        # print('새로운 토큰은 이렇게 됩니다', new_tokens)
        # print('새로운 토큰의 길이', len(new_tokens))

        input_ids = tokenizer.convert_tokens_to_ids(new_tokens)
        input_mask = [1] * len(input_ids)

        features.append(
            InputFeatures(
                tokens=new_tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
        # print('토큰마지막은 이렇게 됩니다', features)
        # print('토큰마지막의 길이', len(features))
    return features
