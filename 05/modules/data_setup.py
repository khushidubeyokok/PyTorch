{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "toc_visible": true,
      "authorship_tag": "ABX9TyPVC126fzzG9QD0jqSQaYDh",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/khushidubeyokok/PyTorch/blob/main/05/modules/data_setup.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "pOg4M1ojIrC-"
      },
      "outputs": [],
      "source": [
        "\"\"\" contains functionality for creating pytorch dataloader for image classification data \"\"\"\n",
        "import os\n",
        "from torch.utils.data import DataLoader\n",
        "from torchvision import datasets,transforms\n",
        "num_workers=os.cpu_count()\n",
        "\n",
        "def create_dataloaders(\n",
        "    train_dir:str,\n",
        "    test_dir:str,\n",
        "    transform:transforms.Compose,\n",
        "    batch_size:int,\n",
        "    num_workers:int=num_workers):\n",
        "  \"\"\" Creates training and testing dataloaders\n",
        "\n",
        "  Takes in a training directory and testing directory path and turns\n",
        "  them into PyTorch Datasets and then into PyTorch DataLoaders.\n",
        "\n",
        "  Args:\n",
        "    train_dir: Path to training directory.\n",
        "    test_dir: Path to testing directory.\n",
        "    transform: torchvision transforms to perform on training and testing data.\n",
        "    batch_size: Number of samples per batch in each of the DataLoaders.\n",
        "    num_workers: An integer for number of workers per DataLoader.\n",
        "\n",
        "  Returns:\n",
        "    A tuple of (train_dataloader, test_dataloader, class_names).\n",
        "    Where class_names is a list of the target classes.\n",
        "    Example usage:\n",
        "      train_dataloader, test_dataloader, class_names = \\\n",
        "        = create_dataloaders(train_dir=path/to/train_dir,\n",
        "                             test_dir=path/to/test_dir,\n",
        "                             transform=some_transform,\n",
        "                             batch_size=32,\n",
        "                             num_workers=4) \"\"\"\n",
        "\n",
        "  # use ImageFolder to create dataset(s)\n",
        "  train_data=datasets.ImageFolder(root=train_dir,transform=transform,target_transform=None)\n",
        "  test_data=datasets.ImageFolder(root=test_dir,transform=transform,target_transform=None)\n",
        "\n",
        "\n",
        "  # get class names as list\n",
        "  class_names=train_data.classes\n",
        "\n",
        "  train_dataloader=DataLoader(dataset=train_data,batch_size=batch_size,num_workers=num_workers,shuffle=True,pin_memory=True)\n",
        "  test_dataloader=DataLoader(dataset=test_data,batch_size=batch_size,num_workers=num_workers,shuffle=True,pin_memory=True)\n",
        "  print(f\"{len(train_dataloader)},{len(test_dataloader)}\")\n",
        "\n",
        "  return train_dataloader,test_dataloader,class_names"
      ]
    }
  ]
}