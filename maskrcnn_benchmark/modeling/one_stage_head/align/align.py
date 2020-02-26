import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.rpn.fcos.fcos import build_fcos

from torch.autograd import Variable

import random

INF = 10000000000


class CTCPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(CTCPredictor, self).__init__()
        self.voc_size = cfg.DATASETS.TEXT.VOC_SIZE
        conv_func = conv_with_kaiming_uniform(True, True, False, False)
        convs = []
        for i in range(2):
            convs.append(conv_func(in_channels, in_channels, 3, stride=(2, 1)))
        self.convs = nn.Sequential(*convs)
        self.rnn = nn.LSTM(in_channels, in_channels,
                           num_layers=1,
                           bidirectional=True)
        self.clf = nn.Linear(in_channels * 2, self.voc_size)

    def forward(self, x, targets=None):
        # average along H dimension
        x = self.convs(x)
        x = x.mean(dim=2)  # NxCxW
        x = x.permute(2, 0, 1)  # WxNxC
        x, _ = self.rnn(x)
        x = self.clf(x)
        if self.training:
            x = F.log_softmax(x, dim=-1)
            input_lengths = torch.full((x.size(1),), x.size(0), dtype=torch.long)
            target_lengths, targets = self.prepare_targets(targets)
            loss = F.ctc_loss(x, targets, input_lengths, target_lengths, blank=self.voc_size-1) / 10
            return loss
        return x
    
    def prepare_targets(self, targets):
        target_lengths = (targets != self.voc_size - 1).long().sum(dim=-1)
        sum_targets = [t[:l] for t, l in zip(targets, target_lengths)]
        sum_targets = torch.cat(sum_targets)
        return target_lengths, sum_targets


class BidirectionalLSTM(nn.Module):
    
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    def __init__(self, cfg, in_channels):
        super(CRNN, self).__init__()
        self.voc_size = cfg.DATASETS.TEXT.VOC_SIZE
        conv_func = conv_with_kaiming_uniform(True, True, False, False)
        convs = []
        for i in range(2):
            convs.append(conv_func(in_channels, in_channels, 3, stride=(2, 1)))
        self.convs = nn.Sequential(*convs)
        self.rnn = BidirectionalLSTM(in_channels,in_channels,in_channels)

    def forward(self, x):
        # average along H dimension
        x = self.convs(x)
        x = x.mean(dim=2)  # NxCxW
        x = x.permute(2, 0, 1)  # WxNxC
        x = self.rnn(x)
        return x


# apply attention
class Attention(nn.Module):
    def __init__(self, cfg, in_channels):
        super(Attention, self).__init__()
        self.hidden_size = in_channels
        self.output_size = cfg.DATASETS.TEXT.VOC_SIZE
        self.dropout_p = 0.1
        self.max_len = cfg.DATASETS.TEXT.NUM_CHARS

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # test
        self.vat = nn.Linear(self.hidden_size, 1)

    def forward(self, input, hidden, encoder_outputs):
        '''
        hidden: 1 x n x self.hidden_size
        encoder_outputs: time_step x n x self.hidden_size (T,N,C)
        '''
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        # test
        batch_size = encoder_outputs.shape[1]

        alpha = hidden + encoder_outputs
        alpha = alpha.view(-1, alpha.shape[-1]) # (T * n, hidden_size)
        attn_weights = self.vat(torch.tanh(alpha))  # (T * n, 1)
        attn_weights = attn_weights.view(-1, 1, batch_size).permute((2,1,0)) # (T, 1, n)  -> (n, 1, T)
        attn_weights = F.softmax(attn_weights, dim=2)

        attn_applied = torch.matmul(attn_weights,
                                    encoder_outputs.permute((1, 0, 2)))

        output = torch.cat((embedded, attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(0) # (1, n, hidden_size)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden) # (1, n, hidden_size)

        output = F.log_softmax(self.out(output[0]), dim=1)  # (n, hidden_size)
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result

    def prepare_targets(self, targets):
        target_lengths = (targets != self.output_size - 1).long().sum(dim=-1)
        sum_targets = [t[:l] for t, l in zip(targets, target_lengths)]
        return target_lengths, sum_targets


class ATTPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(ATTPredictor, self).__init__()
        self.CRNN = CRNN(cfg, in_channels)
        self.criterion = torch.nn.NLLLoss()
        self.attention = Attention(cfg, in_channels)
        self.teach_prob = 1.0

    def forward(self, rois, targets=None):
        rois = self.CRNN(rois)
        if self.training:
            text = targets
            target_variable = text
            _init = torch.zeros((rois.size()[1], 1)).long()
            _init = torch.LongTensor(_init).cuda()
            target_variable = torch.cat((_init, target_variable), 1)
            target_variable = target_variable.cuda()
            decoder_input = target_variable[:,0] # init decoder, from 0
            decoder_hidden = self.attention.initHidden(rois.size()[1]).cuda() # batch rois.size[1]
            loss = 0.0
            try:
                for di in range(1, target_variable.shape[1]):
                    decoder_output, decoder_hidden, decoder_attention = self.attention( #  decoder_output (nbatch, ncls)
                        decoder_input, decoder_hidden, rois)
                    loss += self.criterion(decoder_output, target_variable[:,di])
                    teach_forcing = True if random.random() > self.teach_prob else False
                    if teach_forcing:
                        decoder_input = target_variable[:,di]  # Teacher forcing
                    else:
                        topv, topi = decoder_output.data.topk(1)
                        ni = topi.squeeze()
                        decoder_input = ni
            except Exception as e:
                print(e)
                loss = 0.0
            return loss
        else:
            n = rois.size()[1]
            decodes = torch.zeros((n, self.attention.max_len))
            prob = 1.0
            decoder_input = torch.zeros(n).long().cuda() 
            decoder_hidden = self.attention.initHidden(n).cuda()
            try:
                for di in range(self.attention.max_len):
                    decoder_output, decoder_hidden, decoder_attention = self.attention(
                        decoder_input, decoder_hidden, rois)
                    probs = torch.exp(decoder_output)
                    topv, topi = decoder_output.data.topk(1)
                    ni = topi.squeeze()
                    decoder_input = ni
                    prob *= probs[:, ni]
                    decodes[:, di] = decoder_input.clone()
                decodes = torch.as_tensor(decodes).cuda()
            except:
                decodes += 96
            return decodes


class AlignHead(nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(AlignHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        resolution = cfg.MODEL.ALIGN.POOLER_RESOLUTION
        canonical_scale = cfg.MODEL.ALIGN.POOLER_CANONICAL_SCALE

        self.scales = cfg.MODEL.ALIGN.POOLER_SCALES
        self.pooler = Pooler(
            output_size=resolution,
            scales=self.scales,
            sampling_ratio=1,
            canonical_scale=canonical_scale,
            mode='bezier')

        for head in ['rec']:
            tower = []
            conv_block = conv_with_kaiming_uniform(
                True, True, False, False)
            for i in range(cfg.MODEL.ALIGN.NUM_CONVS):
                tower.append(
                    conv_block(in_channels, in_channels, 3, 1))
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))
        
        self.predict_type = cfg.MODEL.ALIGN.PREDICTOR
        if self.predict_type == "ctc":
            self.predictor = CTCPredictor(cfg, in_channels)
        elif self.predict_type == "attention":
            self.predictor = ATTPredictor(cfg, in_channels)
        else:
            raise("Unknown recognition predictor.")

    def forward(self, x, proposals):
        """
        offset related operations are messy
        """
        beziers = [p.get_field("beziers") for p in proposals]
        rois = self.pooler(x, proposals, beziers)
        rois = self.rec_tower(rois)

        if self.training:
            targets = []
            for proposals_per_im in proposals:
                targets.append(proposals_per_im.get_field("rec").rec)
            targets = torch.cat(targets, dim=0)

            loss = self.predictor(rois, targets)
                        
            return None, loss
        else:
            # merge results and preds
            if self.predict_type == "ctc":
                logits = self.predictor(rois)
                _, preds = logits.permute(1, 0, 2).max(dim=-1)
            elif self.predict_type == "attention":
                preds = self.predictor(rois)

            start_ind = 0
            for proposals_per_im in proposals:
                end_ind = start_ind + len(proposals_per_im)
                proposals_per_im.add_field("recs", preds[start_ind:end_ind])
                start_ind = end_ind

            return proposals, {}


class AlignModule(torch.nn.Module):
    """
    Module for BezierAlign computation. Takes feature maps from the backbone and
    BezierAlign outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(AlignModule, self).__init__()

        self.cfg = cfg.clone()
        self.head = AlignHead(cfg, in_channels)
        self.detector = build_fcos(cfg, in_channels)
        self.scales = cfg.MODEL.ALIGN.POOLER_SCALES

    def forward(self, images, features, targets=None, vis=False):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)
            vis (bool): visualise offsets

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        boxes, losses = self.detector(images, features[1:], targets)

        rec_features = features[:len(self.scales)]

        if self.training:
            _, mask_loss = self.head(rec_features, targets)
            losses.update({'rec_loss': mask_loss})
            return None, losses

        preds, _ = self.head(rec_features, boxes)
        return preds, {}


@registry.ONE_STAGE_HEADS.register("align")
def build_align_head(cfg, in_channels):
    return AlignModule(cfg, in_channels)
