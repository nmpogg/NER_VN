import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pyvi import ViTokenizer
import re
from .format import DateExtractor



# Định nghĩa lớp mô hình NER (phải giống khi huấn luyện)
class PhoBERTForNER(nn.Module):
    def __init__(self, num_labels):
        super(PhoBERTForNER, self).__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        self.dropout = torch.nn.Dropout(0.3)  # Tăng dropout
        self.classifier = torch.nn.Linear(self.phobert.config.hidden_size, num_labels)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            # Chỉ tính loss trên các token thực tế (không phải padding hoặc token đặc biệt)
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, logits.size(-1))[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


def tokenize_vi_full(text):
    # Bước 1: Tách từ bằng pyvi
    tokenized = ViTokenizer.tokenize(text)
    
    # Bước 2: Bỏ dấu _ và tách từng từ
    words = []
    for word in tokenized.split():
        if "_" in word:
            words.extend(word.split("_"))
        else:
            words.append(word)
    
    # Bước 3: Tách dấu câu thành token riêng
    tokens = []
    for word in words:
        # Tách các dấu câu ra khỏi từ (giữ lại dấu câu)
        tokens.extend(re.findall(r"\w+|[^\w\s]", word, re.UNICODE))
    
    return tokens

# Hàm dự đoán NER (hỗ trợ chuỗi đơn hoặc danh sách câu)
def predict_ner(model, tokenizer, sentence, device, id2tag, max_len=128):
    model.eval()

    # Tách từ sử dụng pyvi
    words = tokenize_vi_full(sentence)

    # Tạo đặc trưng
    features = [tokenizer.cls_token_id]
    word_ids = [None]

    for word_idx, word in enumerate(words):
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            print(f"Warning: Word '{word}' could not be tokenized")
            continue
        token_ids = tokenizer.convert_tokens_to_ids(word_tokens)
        features.extend(token_ids)
        word_ids.extend([word_idx] * len(token_ids))

    features.append(tokenizer.sep_token_id)
    word_ids.append(None)

    # Cắt bớt nếu quá dài
    if len(features) > max_len:
        print(f"Warning: Input sentence truncated to {max_len} tokens")
        features = features[:max_len]
        word_ids = word_ids[:max_len]

    # Padding
    attention_mask = [1] * len(features)
    padding_length = max_len - len(features)
    if padding_length > 0:
        features.extend([tokenizer.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        word_ids.extend([None] * padding_length)

    # Chuyển sang tensor
    input_ids = torch.tensor([features], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)

    # Dự đoán
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        predictions = torch.argmax(logits, dim=2).cpu().numpy()[0]

    # Ánh xạ nhãn về từ
    results = []
    prev_word_idx = None
    token_tags = []

    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != prev_word_idx:
            if prev_word_idx is not None:
                tag = token_tags[0]  # Chọn nhãn của token con đầu tiên
                results.append((words[prev_word_idx], tag))
            token_tags = [id2tag[predictions[token_idx]]]
            prev_word_idx = word_idx
        else:
            token_tags.append(id2tag[predictions[token_idx]])

    if prev_word_idx is not None:
        tag = token_tags[0]
        results.append((words[prev_word_idx], tag))

    return results


def main():
    # Thiết lập
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    # Định nghĩa tag2id và id2tag (phải giống khi huấn luyện)
    NER_LABELS = ['O',
              'B-PERSON_NAME', 'I-PERSON_NAME',
              'B-ORGANIZATION', 'I-ORGANIZATION',
              'B-LOCATION', 'I-LOCATION',
              'B-DATE', 'I-DATE',
              'B-PATIENT_ID', 'I-PATIENT_ID',
              'B-GENDER', 'I-GENDER',
              'B-OCCUPATION', 'I-OCCUPATION',
              'B-SYMPTOM_AND_DISEASE', 'I-SYMPTOM_AND_DISEASE',
              'B-TRANSPORTATION', 'I-TRANSPORTATION',
              'B-AGE', 'I-AGE',
              'B-JOB', 'I-JOB',
              'B-MISC', 'I-MISC'
              ]

    # Ánh xạ nhãn sang id và ngược lại
    tag2id = {tag: id for id, tag in enumerate(NER_LABELS)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    # Tải mô hình
    model = PhoBERTForNER(num_labels=len(NER_LABELS))
    model_path = "../models/phobert_ner_model.pth"  # Đường dẫn tới file mô hình
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Test với một câu
    test_text = "Thứ sáu tuần sau Apple tung ra sản phẩm mới."
    tagged_words = predict_ner(model, tokenizer, test_text, device, id2tag)
    extractor = DateExtractor()
    results, changes = extractor.process_ner_results(tagged_words)

    print("\nKết quả nhận diện thực thể:")
    print("Tagged words:", results)

    print("\nSau khi định dạng:")
    print(changes)
    

if __name__ == "__main__":
    main()