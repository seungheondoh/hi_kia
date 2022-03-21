import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Model

class AudioModel(nn.Module):
    def __init__(self, data_type, feature_type, freeze_type):
        super().__init__()
        self.data_type = data_type
        self.feature_type = feature_type
        self.freeze_type = freeze_type
        self.wav2vec = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
        if freeze_type == "feature":
            self.wav2vec.feature_extractor._freeze_parameters()
        elif freeze_type =="encoder":
            for param in self.wav2vec.encoder.parameters():
                param.requires_grad = False
        elif freeze_type == "all_model":
            for param in self.wav2vec.parameters():
                param.requires_grad = False
        self.feature_dim = 768
        self.projector = nn.Linear(self.feature_dim, self.feature_dim)
        self.classifier = nn.Linear(self.feature_dim, 4)
        # self.classifier = nn.Linear(self.feature_dim, 7)
        self.dropout = nn.Dropout(0.5)

    def forward(self, wav, mask=None):
        feature = self.wav2vec(
            wav,
            attention_mask=mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
            )
        last_hidden = feature['last_hidden_state']
        if self.freeze_type == "all_model":
            last_hidden = last_hidden.detach()
        last_hidden = self.projector(last_hidden) # use last hidden layer
        projection_feature = last_hidden.mean(1,False) # temproal average pooling
        dropout_feature = self.dropout(projection_feature)
        logit = self.classifier(dropout_feature)
        return logit, projection_feature