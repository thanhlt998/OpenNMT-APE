from pytorch_pretrained_bert import BertModel

from onmt.encoders.encoder import EncoderBase


class BERTEncoder(EncoderBase):
    """
    Returns:
        (`FloatTensor`, `FloatTensor`):

        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def forward(self, src, lengths=None):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(src, lengths)
        encoded_layers, pooled_output = \
            self.bert(src[:, :, 0].t(), output_all_encoded_layers=False)

        return pooled_output.unsqueeze(0),\
            encoded_layers.transpose(0, 1), lengths

    def initialize_bert(self, bert_type, checkpoint=None):

        def _remove_prefix(component_name):
            '''
            remove the first part of the name of a component of the state dict.
            In this case, we're removing "encoder." from the component name so
            Bert can be loaded by the .from_pretrained() method.
            '''

            return '.'.join(component_name.split('.')[1:])

        if checkpoint:
            bert_state_dict = {
                _remove_prefix(k): v for k, v in checkpoint['model'].items()
                if 'bert' in k}
        else:
            bert_state_dict = None
        self.bert = BertModel.from_pretrained(bert_type,
                                              bert_state_dict)