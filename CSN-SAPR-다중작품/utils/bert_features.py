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
    tokens_list = []
    # recognize candidate and save as token
    # 발화자를 인식하고 토큰으로 저장합니다.
    for (ex_index, example) in enumerate(examples):
        tokens = tokenizer.tokenize(example)
        tokens_list.append(tokens)

        new_tokens = []
        input_type_ids = []

        new_tokens.append("[CLS]")
        input_type_ids.append(0)
        new_tokens += tokens
        input_type_ids += [0] * len(tokens)
        new_tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(new_tokens)
        input_mask = [1] * len(input_ids)

        features.append(
            InputFeatures(
                tokens=new_tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features, tokens_list
