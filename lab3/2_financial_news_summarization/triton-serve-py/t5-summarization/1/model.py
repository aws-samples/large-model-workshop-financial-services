import json
import logging
import subprocess
import sys

import torch
import transformers
import numpy as np 
import triton_python_backend_utils as pb_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonPythonModel:

    def initialize(self, args):
        self.model_dir = args['model_repository']
        
        # Workaround for Triton model repo naming bug
        if 'model.py' in self.model_dir:
            dir_list = self.model_dir.split(sep='/')[1:-2]
            self.model_dir = ''
            for dirname in dir_list:
                self.model_dir += f'/{dirname}'

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(f'{self.model_dir}/tokenizer')
        self.device_id = args['model_instance_device_id']
        self.device = torch.device(f'cuda:{self.device_id}') if torch.cuda.is_available() else torch.device('cpu')
        self.model = transformers.T5ForConditionalGeneration.from_pretrained(f'{self.model_dir}/model').eval().to(self.device)
        self.model_config = model_config = json.loads(args['model_config'])

        output0_config = pb_utils.get_output_config_by_name(
            model_config, "SUMMARY")

        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        
    def execute(self, requests):
        
        file = open("logs.txt", "w")
        responses = []
        for request in requests:
            logger.info("Request: {}".format(request))
            
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in_0 = in_0.as_numpy()
            
            logger.info("in_0: {}".format(in_0))
                        
            tokenized_batch = []
            for i in range(in_0.shape[0]):                
                decoded_text = in_0[i,0].decode()
                
                logger.info("decoded_object: {}".format(decoded_text))
                                        
                prepared_text = 'summarize: ' + decoded_text
                tokenized_batch.append(decoded_text)
                
            logger.info("tok_batch: {}".format(tokenized_batch))
            
            input_ids = self.tokenizer(tokenized_batch,
                                       padding=True,
                                       pad_to_multiple_of=8,
                                       max_length=512,
                                       truncation=True,
                                       return_tensors='pt'
                                      )['input_ids'].to(self.device)
            
            with torch.no_grad():
                with torch.autocast(device_type='cuda',enabled=True):
                    model_output = self.model.generate(input_ids,
                                                       num_beams=4,
                                                       no_repeat_ngram_size=2,
                                                       min_length=30,
                                                       max_length=50,
                                                       early_stopping=True)
                        
                    summaries = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)

            batched_response = [[summaries[i]] for i in range(len(summaries))]
            out0 = np.array(batched_response, dtype=self.output0_dtype)
            logger.info("out_0: {}".format(out0))
            
            out_tensor_0 = pb_utils.Tensor("SUMMARY", out0)
            logger.info("out_tensor_0: {}".format(out_tensor_0))
            
            responses.append(pb_utils.InferenceResponse([out_tensor_0]))
            
        return responses