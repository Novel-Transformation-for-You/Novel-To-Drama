# CSN module definition
import torch.nn as nn
import torch.nn.functional as functional
import torch
from transformers import AutoModel

def get_nonlinear(nonlinear):
    """
    Activation function.
    """
    nonlinear_dict = {'relu':nn.ReLU(), 'tanh':nn.Tanh(), 'sigmoid':nn.Sigmoid(), 'softmax':nn.Softmax(dim=-1)}
    try:
        return nonlinear_dict[nonlinear]
    except:
        raise ValueError('not a valid nonlinear type!')


class SeqPooling(nn.Module):
    """
    Sequence pooling module.

    Can do max-pooling, mean-pooling and attentive-pooling on a list of sequences of different lengths.
    """
    def __init__(self, pooling_type, hidden_dim):
        super(SeqPooling, self).__init__()
        self.pooling_type = pooling_type
        self.hidden_dim = hidden_dim
        if pooling_type == 'attentive_pooling':
            self.query_vec = nn.parameter.Parameter(torch.randn(hidden_dim))

    def max_pool(self, seq):
        return seq.max(0)[0]
        print(seq)
        return seq.max(0)

    def mean_pool(self, seq):
        return seq.mean(0)

    def attn_pool(self, seq):
        attn_score = torch.mm(seq, self.query_vec.view(-1, 1)).view(-1)
        attn_w = nn.Softmax(dim=0)(attn_score)
        weighted_sum = torch.mm(attn_w.view(1, -1), seq).view(-1)     
        return weighted_sum

    def forward(self, batch_seq):
        pooling_fn = {'max_pooling': self.max_pool,
                      'mean_pooling': self.mean_pool,
                      'attentive_pooling': self.attn_pool}
        # print(seq)
        pooled_seq = [pooling_fn[self.pooling_type](seq) for seq in batch_seq]
        return torch.stack(pooled_seq, dim=0)


class MLP_Scorer(nn.Module):
    """
    MLP scorer module.

    A perceptron with two layers.
    """
    def __init__(self, args, classifier_input_size):
        super(MLP_Scorer, self).__init__()
        self.scorer = nn.ModuleList()

        self.scorer.append(nn.Linear(classifier_input_size, args.classifier_intermediate_dim))
        self.scorer.append(nn.Linear(args.classifier_intermediate_dim, 1))
        self.nonlinear = get_nonlinear(args.nonlinear_type)

    def forward(self, x):
        for model in self.scorer:
            x = self.nonlinear(model(x))
        return x


class CSN(nn.Module):
    """
    Candidate Scoring Network.

    It's built on BERT with an MLP and other simple components.
    """
    def __init__(self, args):
        super(CSN, self).__init__()
        self.args = args
        self.bert_model = AutoModel.from_pretrained(args.bert_pretrained_dir)
        self.pooling = SeqPooling(args.pooling_type, self.bert_model.config.hidden_size)
        self.mlp_scorer = MLP_Scorer(args, self.bert_model.config.hidden_size * 3)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, features, sent_char_lens, mention_poses, quote_idxes, true_index, device, tokens_list, cut_css):
        """
        params
            features: the candidate-specific segments (CSS) converted into the form of BERT input.  
            sent_char_lens: character-level lengths of sentences in CSSs.
                [[character-level length of sentence 1,...] in the CSS of candidate 1,...]
            mention_poses: the positions of the nearest candidate mentions.
                [(sentence-level index of nearest mention in CSS, 
                 character-level index of the leftmost character of nearest mention in CSS, 
                 character-level index of the rightmost character + 1) of candidate 1,...]
            quote_idxes: the sentence-level index of the quotes in CSSs.
                [index of quote in the CSS of candidate 1,...]
            true_index: the index of the true speaker.
            device: gpu/tpu/cpu device.
        """
        # encoding
        qs_hid = []
        ctx_hid = []
        cdd_hid = []
        # print('첫번째 인풋', sent_char_lens)
        # print('두번째 인풋', mention_poses)
        # print('세번째 인풋', quote_idxes)
        
        for i, (cdd_sent_char_lens, cdd_mention_pos, cdd_quote_idx) in enumerate(zip(sent_char_lens, mention_poses, quote_idxes)):
            # print(f'{i}번째 ################################################################입니다')
            # print('네번째 아웃풋', cdd_sent_char_lens)
            # print('다섯번째 아웃풋', cdd_mention_pos)
            # print('여섯번째 아웃풋', cdd_quote_idx)

            # print('bert 인풋 확인', [features[i].input_ids], [features[i].input_mask])
            bert_output = self.bert_model(torch.tensor([features[i].input_ids], dtype=torch.long).to(device), token_type_ids=None, 
                attention_mask=torch.tensor([features[i].input_mask], dtype=torch.long).to(device))
            # print('bert 결과 확인', bert_output)
            # print('bert 결과 확인', bert_output['last_hidden_state'])
            # print('bert size', bert_output['last_hidden_state'].shape)
            # print('tokens_list 입니다.', tokens_list[i])
            # modify_list = tokens_list[i]
            modified_list = [s.replace('#', '') for s in tokens_list[i]]

            # print('token_list len', len(tokens_list[i]))

            # print('드디어 여기서 cut_css', cut_css)
            # print('len cut_css', len(cut_css))
            # print('원본', cut_css)
            # print('순서에 맞게 봅시다', cut_css[i])

            import re
            cnt = 1
            verify = 0
            verify_name = 0
            accum_char_len = [0]
            # print('몇번까지 돌까요?', len(cut_css[i]))
            for idx in cut_css[i]:
                result_string = ''.join(idx)
                # string = result_string[-5:]
                string1 = result_string[-5:].replace(']', '\]')
                string2 = string1.replace('[', '\[')
                string3 = string2.replace('?', '\?')
                # string.replace('[', '\[')
                # string.replace('?', '\?')
                    
                pattern = re.compile(rf'[{string3}]')
                # print(pattern)
                # print('오호...')
                cnt = 1
                for string in modified_list:
                    # print(string)
                    string_nospace = string.replace(' ','')
                    for letter in string_nospace:
                        match_result = pattern.match(letter)

                        if match_result:
                            end_index = match_result.end()
                            verify += 1
                            # print(letter, verify)
                            if verify == len(result_string[-5:]):
                                # print(cnt,letter, verify)
                                accum_char_len.append(cnt)
                                verify = 0
                        else:
                            verify = 0
                            # print('못찾았습니다', letter)
                    cnt += 1
                
            # for sent_idx in range(len(cdd_sent_char_lens)):
            #     # print('전', accum_char_len)
            #     print('전2', cdd_sent_char_lens)
            #     # print('전3', sent_idx)
            #     accum_char_len.append(accum_char_len[-1] + cdd_sent_char_lens[sent_idx])
            #     print('후', accum_char_len)
            
            # print('CSS_hid을 한번 봅시다 1', CSS_hid)
            # print(features, features[0])
            CSS_hid = bert_output['last_hidden_state'][0][1:sum(cdd_sent_char_lens) + 1]
            # print('CSS_hid을 한번 봅시다 2', CSS_hid)
            # print(CSS_hid)
            # print(type(CSS_hid))
            # print(len(CSS_hid))
            # print('몇차원 일까요? - 1번', CSS_hid.shape)
            # print('qs hid을 한번 봅시다 1', qs_hid)
            # print(accum_char_len[cdd_quote_idx], accum_char_len[cdd_quote_idx + 1])
            # print('여기서 찾아봐야 합니다!! - accum_char_len', accum_char_len )
            # print('여기서 찾아봐야 합니다!!22 - cdd_quote_idx', cdd_quote_idx )
            qs_hid.append(CSS_hid[accum_char_len[cdd_quote_idx]:accum_char_len[cdd_quote_idx + 1]])
            # print('qs hid을 한번 봅시다 2', qs_hid)
            # print(qs_hid)
            # print(type(qs_hid))
            # print('몇차원 일까요? - 2번', len(qs_hid))

            ## 발화자 부분 찾아서 - bert tokenizer 된 부분을 인덱싱 하는 부분
            cnt = 1
            cdd_mention_pos_bert_li = []
            name = cut_css[i][cdd_mention_pos[0]][cdd_mention_pos[3]]
            pattern_name = re.compile(rf'[{name}]')
            # print('처음@@@', accum_char_len)
            # print(cdd_sent_char_lens)
            # print('끝@@@@', cdd_mention_pos[0])
            if len(accum_char_len) < cdd_mention_pos[0]+1:
                maxx_len = accum_char_len[len(accum_char_len)-1] 
            elif len(accum_char_len) == cdd_mention_pos[0]+1:
                maxx_len = accum_char_len[-1] + 1000 
            else:
                maxx_len = accum_char_len[cdd_mention_pos[0]+1] 
            # print('max', maxx_len)
            for string in modified_list:
                string_nospace = string.replace(' ','')
                for letter in string_nospace:
                    match_result_name = pattern_name.match(letter)
                    if match_result_name and maxx_len > cnt >= accum_char_len[cdd_mention_pos[0]]:
                        verify_name += 1
                        # print(letter, cnt)
                        # print(match_result_name)
                        # print(verify_name)
                        # print('전', cdd_mention_pos_bert_li)
                        if len(cdd_mention_pos_bert_li) < 2:
                            if verify_name == 1:
                                cdd_mention_pos_bert_li.append(cnt)
                                temp_cnt = 1
                                # print(cdd_mention_pos_bert_li)
                            elif verify_name == len(name):
                                cdd_mention_pos_bert_li.append(cnt)
                                verify_name = 0
                            # print('후1', cdd_mention_pos_bert_li)
                    elif verify_name == 1 and len(cdd_mention_pos_bert_li)>0 and len(cdd_mention_pos_bert_li) != 2:
                        cdd_mention_pos_bert_li = cdd_mention_pos_bert_li[:-1]
                        # print('후2', cdd_mention_pos_bert_li)
                        verify_name = 0
                    else:
                        verify_name = 0
                cnt += 1

            # accum_char_len = [0]
            # print('cdd_sent_char_lens', cdd_sent_char_lens)
            # print('cdd_quote_idx', cdd_quote_idx)
            # print('accum_char_len', accum_char_len)

            if len(cdd_sent_char_lens) == 1:
                ctx_hid.append(torch.zeros(1, CSS_hid.size(1)).to(device))
            elif cdd_mention_pos[0] == 0:
                ctx_hid.append(CSS_hid[:accum_char_len[-2]])
            else:
                ctx_hid.append(CSS_hid[accum_char_len[1]:])
            
            cdd_mention_pos_bert = (cdd_mention_pos[0], cdd_mention_pos_bert_li[0], cdd_mention_pos_bert_li[1])
            # print('cdd_mention_pos', cdd_mention_pos)
            # print('cdd_mention_pos_bert', cdd_mention_pos_bert)
            # print('몇차원 일까요? - 1번', CSS_hid.shape)
            # print('name', name)
            # print('발화자 찾아보자', cut_css[i][cdd_mention_pos_bert[0]][cdd_mention_pos_bert[3]])
            cdd_hid.append(CSS_hid[cdd_mention_pos_bert[1]:cdd_mention_pos_bert[2]])
            # print('cdd_hid', cdd_hid) 

        # pooling
        # print(qs_hid, ctx_hid, cdd_hid)
        qs_rep = self.pooling(qs_hid)
        ctx_rep = self.pooling(ctx_hid)
        # print('cdd_rep', cdd_hid)
        cdd_rep = self.pooling(cdd_hid)

        # concatenate
        feature_vector = torch.cat([qs_rep, ctx_rep, cdd_rep], dim=-1)

        # dropout
        feature_vector = self.dropout(feature_vector)
        
        # scoring
        scores = self.mlp_scorer(feature_vector).view(-1)
        scores_false = [scores[i] for i in range(scores.size(0)) if i != true_index]
        scores_true = [scores[true_index] for i in range(scores.size(0) - 1)]

        return scores, scores_false, scores_true

        