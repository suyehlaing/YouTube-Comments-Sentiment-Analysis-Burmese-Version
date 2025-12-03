from transformers import pipeline

# Initialize the pipeline with the model
sentiment_pipeline = pipeline(
    "zero-shot-classification", 
    model="joeddav/xlm-roberta-large-xnli"
)

# Save the model locally
sentiment_pipeline.model.save_pretrained("./local_model")
sentiment_pipeline.tokenizer.save_pretrained("./local_model")
print("Model loaded and saved successfully.")