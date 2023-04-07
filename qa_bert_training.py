# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 01:06:46 2023

@author: Md Mamunur Rahman, Data has been taken from Krishnaik
"""
if __name__ == '__main__':   
    import json
    import logging
    from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs 
    
    ### Data uploading #####################################
    with open(r"data/train.json", "r") as read_file:
              training_data = json.load(read_file)
    #print (training_data)
    
    with open(r"data/test.json", "r") as read_file:
              testing_data = json.load(read_file)
    #print (testing_data)
    
    with open(r"data/predictions.json", "r") as read_file:
              predictions_data = json.load(read_file)
    #print (predictions_data)
    ########################################################
    
    ### Model construction #################################
    
    # pretrained model inclusion 
    model_type = "bert"
    model_name = "bert-base-cased"
    
    # Configure the model 
    model_args = QuestionAnsweringArgs()
    model_args.train_batch_size = 16
    model_args.evaluate_during_training = True
    model_args.n_best_size=3
    model_args.num_train_epochs=5
    
    #setup QA arguments
    train_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "use_cached_eval_features": True,
        "output_dir": f"outputs/{model_type}",
        "best_model_dir": f"outputs/{model_type}/best_model",
        "evaluate_during_training": True,
        "max_seq_length": 128,
        "num_train_epochs": 15,
        "evaluate_during_training_steps": 1000,
        #"wandb_project": "Question Answer Application",
        #"wandb_kwargs": {"name": model_name},
        "save_model_every_epoch": False,
        "save_eval_checkpoints": False,
        "n_best_size":3,
        # "use_early_stopping": True,
        # "early_stopping_metric": "mcc",
        # "n_gpu": 2,
         "manual_seed": 4,
         #"use_multiprocessing": False,
        "train_batch_size": 2,
        "eval_batch_size": 2,
        # "config": {
        #     "output_hidden_states": True
        # }
    }
    
    #initialize the model
    model = QuestionAnsweringModel(
        model_type, model_name, args=train_args, use_cuda=False #if there is no GPU use_cuda=False
        )
    
    #train the model
    model.train_model(training_data, eval_data=testing_data)
    
    #evaluate model
    result, texts = model.eval_model(testing_data)
    print(result)
    print(texts)
    
    #prediction with model
    to_predict = [
    {
        "context": "Mamun is a software developer and Mamun lives in Toronto",
        "qas": [
            {
                "question": "Who is Mamun?",
                "id": "0",
            }
            ],
        }
    ]
    result, texts = model.predict(to_predict)
    print(result)
    print(texts)