from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from test import PhoBERTForNER, predict_ner

import format

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Định nghĩa model cho request
class TextRequest(BaseModel):
    text: str

# Khởi tạo model và tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# Định nghĩa nhãn NER
NER_LABELS = ['O',
              'B-NAME', 'I-NAME',
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
              ]

# Ánh xạ nhãn
tag2id = {tag: id for id, tag in enumerate(NER_LABELS)}
id2tag = {id: tag for tag, id in tag2id.items()}

# Tải mô hình
model = PhoBERTForNER(num_labels=len(NER_LABELS))
model_path = "models/phobert_ner_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

@app.post("/api/ner")
async def process_text(request: TextRequest):
    try:
        # Dự đoán NER
        tagged_words = predict_ner(model, tokenizer, request.text, device, id2tag)
        extractor = format.DateExtractor()
        tagged_words_after_format = extractor.process_ner_results(tagged_words)
        
        # Chuyển đổi kết quả sang định dạng phù hợp với giao diện
        entities = []
        current_entity = None
        
        for word, tag in tagged_words_after_format:
            if tag.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'text': word,
                    'label': tag[2:],  # Bỏ prefix B-
                    'start': request.text.find(word),
                    'end': request.text.find(word) + len(word)
                }
            elif tag.startswith('I-') and current_entity:
                current_entity['text'] += ' ' + word
                current_entity['end'] = request.text.find(word) + len(word)
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
            
        return {"entities": entities}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 