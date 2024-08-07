# Configuration

def get_config(num_epochs, d_model, h, depth, num_classes):
  return{
      "train_batch_size": 8,
      "test_batch_size": 1,
      "num_epochs": num_epochs,
      "lr": 3e-4,
      "seq_len": 256,
      "d_model": d_model,
      "h": h,
      "depth" : depth,
      "dropout": 0.2,
      "num_classes": num_classes,
      "checkpoint": 'bert-base-uncased'
  }
