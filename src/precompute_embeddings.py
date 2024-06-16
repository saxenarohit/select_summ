import os
from transformers import RobertaTokenizer,RobertaModel,BartTokenizer,BartModel,LEDTokenizer, LEDModel
import pickle
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from tqdm import tqdm
from datasets import load_dataset
import argparse

def savePickle(path,data):
    with open(path,"wb") as f:
        pickle.dump(data,f)

def load_model():


    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    model = RobertaModel.from_pretrained("roberta-large")

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return tokenizer,model


def loadData():

    mensa_dataset = load_dataset("rohitsaxena/MENSA")   

    return mensa_dataset["train"], mensa_dataset["validation"], mensa_dataset["test"]

def sceneTextToEmbed(model,tokenizer,data):

    processed = []

    for idx in tqdm(range(len(data))):
        movieData = data[idx]

        with torch.no_grad():
            encoded_input = tokenizer(movieData["scenes"], return_tensors='pt', padding=True, truncation=True)
            output = model(**encoded_input.to(device))
            emneddings = output.last_hidden_state[:, 0, :]
            emneddings = emneddings.detach().cpu()

        movieData["scenes_embeddings"] = emneddings
        movieData["labels"] = torch.tensor(movieData["labels"])
        processed.append(movieData)
        
        
    return processed

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description="Evaluate models on Mini-MMLU dataset")
    parser.add_argument(
        "--output_path_root",
        type=str,
        default = "./outputs/",
        choices=["gpt4", "gpt-4-turbo", "llama", "claude"],
        help="Path to extract embedding to be used for classification model",
    )
    
    args = parser.parse_args()
      
    print("Extracting embeddings")

    
    trainData, valData, testData = loadData()
    tokenizer, model = load_model()
    newTrain = sceneTextToEmbed(model,tokenizer,trainData)
    newVal = sceneTextToEmbed(model,tokenizer,valData)
    newTest = sceneTextToEmbed(model,tokenizer,testData)
    

    output_path = os.path.join(args.output_path_root,"scene_classification_data")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    savePickle(os.path.join(output_path, "train.pkl"), newTrain)
    savePickle(os.path.join(output_path, "val.pkl"), newVal)
    savePickle(os.path.join(output_path, "test.pkl"), newTest)  