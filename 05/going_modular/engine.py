'''contains functions for training and testing a pytorch model'''
import torch
from typing import Dict,List,Tuple
device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model,dataloader,loss_function,optimiser,device=device):
  model=model.to(device)
  model.train()
  train_loss=0
  train_acc=0
  for batch,(X,y) in enumerate(dataloader):
    X,y=X.to(device),y.to(device)
    y_pred=model(X) #output model logits
    loss=loss_function(y_pred,y)
    train_loss+=loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    #calculate accuracy
    y_pred_class=torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
    train_acc+=(y_pred_class==y).sum().item()/len(y_pred)

  train_loss=train_loss/len(dataloader)
  train_acc=train_acc/len(dataloader)

  return train_loss,train_acc

def test_model(model,dataloader,loss_function,optimiser,device=device):
  model=model.to(device)
  model.eval()
  test_loss,test_acc=0,0
  with torch.inference_mode():
    for batch,(X,y) in enumerate(dataloader):
      X,y=X.to(device),y.to(device)
      test_pred_logits=model(X)
      loss=loss_function(test_pred_logits,y)
      test_loss+=loss.item()
      test_pred_class=torch.argmax(torch.softmax(test_pred_logits,dim=1),dim=1)
      test_acc+=(test_pred_class==y).sum().item()/len(test_pred_class)

    test_loss=test_loss/len(dataloader)
    test_acc=test_acc/len(dataloader)

  return test_loss,test_acc

def train_and_test_model(model,epochs,train_dataloader,test_dataloader,loss_function,optimiser,device):
  from timeit import default_timer as timer
  from tqdm.auto import tqdm
  start = timer()
  results={"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []}
  for epoch in tqdm(range(epochs)):
    train_loss,train_acc=train_model(model,train_dataloader,loss_function,optimiser,device)
    test_loss,test_acc=test_model(model,test_dataloader,loss_function,optimiser)
    print(f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc*100:.4f}% | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc*100:.4f}%")
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)
  end = timer()
  time_taken=end-start
  print(f"time_taken:{time_taken}")
  return results
