import os
import torch
from transformers import BertModel
from transformers import BertTokenizer

class Model(torch.nn.Module):
    def __init__(self, pretrained_model, tokenizer):
        super().__init__()
        self.fc = torch.nn.Linear(1024, 2)
        self.pretrained = BertModel.from_pretrained(pretrained_model)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        for name, param in self.pretrained.named_parameters():
           if "encoder.layer.11" in name or "pooler" in name:
               param.requires_grad_(False)
           else:
                param.requires_grad_(False)
                # print(f"Parameter name: {name}")
                # print(f"Parameter value: {param}\n")
        # for param in self.pretrained.parameters():
        #     param.requires_grad_(True)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # with torch.no_grad():
        out = self.pretrained(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])
        # out = out.softmax(dim=1)
        return out

    def save(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        torch.save(self.state_dict(), os.path.join(save_directory, 'model_weights.pth'))
        self.pretrained.save_pretrained(os.path.join(save_directory, 'bert_model'))
        self.tokenizer.save_pretrained(os.path.join(save_directory, 'bert_tokenizer'))

        print(f'model save in: {save_directory}')

    @classmethod
    def load(cls, load_directory):
        pretrained_model_path = os.path.join(load_directory, 'bert_model')
        tokenizer_path = os.path.join(load_directory, 'bert_tokenizer')

        print(pretrained_model_path)
        model = cls(pretrained_model=pretrained_model_path, tokenizer=tokenizer_path)

        model_weight_path = os.path.join(load_directory, 'model_weights.pth')
        model.load_state_dict(torch.load(model_weight_path))

        # model.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        print(f'Model loaded from: {load_directory}')

        return model