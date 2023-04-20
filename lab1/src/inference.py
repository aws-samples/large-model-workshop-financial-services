import json
from pathlib import Path
from setfit import SetFitModel

def model_fn(model_dir):
    model = SetFitModel.from_pretrained(model_dir)
    cat_decoder = json.loads(Path(f"{model_dir}/cat_decoder.json").open("r").read())
    
    return model, cat_decoder

def transform_fn(model, input, input_content_type, output_content_type=None):
    model, decoder = model
    if input_content_type != "application/json":
        raise ValueError(f"{input_content_type} is not a supported content type. Use application/json")
    input = json.loads(input)
    predictions = model(input).cpu().numpy()
    predictions = [decoder.get(str(pred)) for pred in predictions]
    
    return json.dumps(predictions)