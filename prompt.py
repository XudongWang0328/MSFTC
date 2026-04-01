import json

with open("./dataset", "r", encoding="utf-8") as f:
    data = json.load(f)
ClassificationPrompt1 = 'This sentence: "*sent_0*" can be classified as:'
ClassificationPrompt2 = 'Classify the following text: "*sent_0*" '
ClassificationPrompt3 = '"*sent_0*" Determine the category of this text.'
ClassificationPrompt4 = 'Which category dose the following text belong to: "*sent_0*" ?'
prompted_data = []
for item in data:
    text = item["text"].strip()
    label = item["label"]
    prompted_text = ClassificationPrompt1.replace("*sent_0*", text)
    prompted_data.append({"prompt_text": prompted_text, "label": label})

with open("./datasetprompt", "w", encoding="utf-8") as f:
    json.dump(prompted_data, f, indent=2, ensure_ascii=False)