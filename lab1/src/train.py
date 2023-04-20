import argparse
import json
from pathlib import Path

import boto3
import datasets
import pandas as pd
import sagemaker
from sagemaker.experiments.run import load_run
from sentence_transformers.losses import CosineSimilarityLoss
from sklearn.metrics import classification_report, confusion_matrix
#SetFit is an efficient and prompt-free framework for few-shot fine-tuning of Sentence Transformers.
#more about setfit: https://github.com/huggingface/setfit
from setfit import SetFitModel, SetFitTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="sentence-transformers/paraphrase-mpnet-base-v2",
        # required=True,
        help="Path to pretrained sent-transformer model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="/opt/ml/input/data/train",
        # required=True,
        help="Path to the train dataset.",
    )

    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="/opt/ml/input/data/test",
        # required=True,
        help="Path to the test dataset.",
    )

    parser.add_argument(
        "--cat_encoders_path",
        type=str,
        default="/opt/ml/input/data/encoders",
        # required=True,
        help="Path to the category encoders.",
    )
    
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")

    parser.add_argument(
        "--num_iterations", type=int, default=20, help="number of training iterations"
    )
    parser.add_argument(
        "--model_dir", type=str, default="/opt/ml/model", help="Model dir."
    )

    args = parser.parse_args()

    return args


def main(args):

    # Get the parameters from the script arguments
    train_path = Path(args.train_dataset_path)
    test_path = Path(args.test_dataset_path)
    encoder_path = Path(args.cat_encoders_path)
    model_id = args.pretrained_model_name_or_path

    # load the datasets and the model
    # SetFitModel: a wrapper that combines a pretrained body from sentence_transformers and
    #  a classification head from either scikit-learn or SetFitHead (a differentiable head built upon PyTorch 
    #  with similar APIs to sentence_transformers).    
    train_ds = datasets.Dataset.from_csv((train_path / "train.csv").as_posix())
    test_ds = datasets.Dataset.from_csv((test_path / "test.csv").as_posix())
    cat_decoder = json.loads(Path(f"{encoder_path}/cat_decoder.json").open("r").read())
    model = SetFitModel.from_pretrained(model_id)

    # create a SetFitTrainer and train the model
    #SetFitTrainer: a helper class that wraps the fine-tuning process of SetFit.
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        loss_class=CosineSimilarityLoss,
        num_iterations=args.num_iterations,
        column_mapping={"text": "text", "label": "label"},
    )

    trainer.train()

    # Evaluate The Model
    predictions = trainer.model(test_ds["text"]).cpu().numpy()
    
    accuracy = trainer.evaluate()["accuracy"]
    
    class_report = classification_report(
        y_true=test_ds["label"],
        y_pred=predictions,
        output_dict=True,
        target_names=cat_decoder.values(),
    )
    Path("/opt/ml/output/class_report.json").open("w").write(json.dumps(class_report))
    
    cm = confusion_matrix(y_true=test_ds["label"], y_pred=predictions)
    df_confusion = pd.melt(
                        pd.DataFrame(
                            cm, index=cat_decoder.values(), columns=cat_decoder.values()
                        ).reset_index(),
                        id_vars="index",
                    )
    

    df_confusion.rename(columns={"index": "actual_category", "variable":"predicted_category"}, inplace=True)
    df_confusion.to_csv("/opt/ml/output/confusion_matrix.csv", index=False)
    
    
    # Log results of Evaluation to SageMaker Experiments
    session = sagemaker.session.Session(boto3.session.Session(region_name=args.region))
    with load_run(sagemaker_session=session) as run:
        # log parameters
        run.log_parameter("model_id", model_id)
        run.log_parameter("num_iterations", trainer.num_iterations)
        
        # log metrics
        run.log_metric("accuracy", accuracy)
        for metric_name, value in class_report["weighted avg"].items():
            run.log_metric(metric_name, value)
        
        # log confusion matrix and classification report
        run.log_file(file_path="/opt/ml/output/class_report.json", name="Classification Report")
        run.log_file(file_path="/opt/ml/output/confusion_matrix.csv", name="Confusion Matrix") 
        

    # Save the model
    trainer.model._save_pretrained(args.model_dir)

    # Save the category decoder so we can use it later during inference
    Path(f"{args.model_dir}/cat_decoder.json").open("w").write(json.dumps(cat_decoder))

if __name__ == "__main__":
    args = parse_args()
    main(args)
