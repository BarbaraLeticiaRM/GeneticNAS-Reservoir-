import torch
from sklearn.metrics import classification_report
from data import get_dataset
from search_config import args

def evaluate_model(model_path, dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _ , testloader, _ = get_dataset({'dataset_name': dataset_name, 'data_path': './data'})

    model = torch.load(model_path, map_location=device, weights_only=False)
    
    model.to(device)
    model.eval()  

    all_labels = []
    all_preds = []

    with torch.no_grad(): # 
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs[0], 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n--- Relatório de Classificação Detalhado por Classe ---")
    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

if __name__ == '__main__':
    evaluate_model(args.model_dir, args.dataset_name)