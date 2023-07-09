import os
import json
import argparse
from src.model import infere_model
import torch

class SummarizationModel:
    def __init__(self,args):
        self.batch_size=args.batch_size # should be 1 for the evaluation script
        self.model_dir = args.model_dir
        self.leader_codalab_email = args.leader_codalab_email
        self.result_save_path = args.result_save_path
        self.model_path=os.path.join(self.model_dir,self.leader_codalab_email)
        self.results={} #List of all jsons of results
        self.data_path=args.data_path
        self.is_val= args.val
        self.data={}
        use_cuda = torch.cuda.is_available()
        self.device= torch.device("cuda" if use_cuda else "cpu")
        print("Device: ",self.device)
        
        assert self.batch_size==1,"batch size must be one"
        assert ("/mnt/" in self.model_path),"model path provided is in incorrect formate"
        assert ("/mnt/" in self.result_save_path),"save path provided is in incorrect formate"
        assert ("/mnt/" in self.data_path),"data path provided is in incorrect formate"

    def get_data(self):
        with open(self.data_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                self.data[int(obj["example_id"])]=obj["paragraph"]

        assert len(list(self.data.keys())) > 0 ,"The jsonl file provided is empty"
        assert all(isinstance(i, str) for i in self.data.values()), "Make sure summary elements of the test data are of type str"
        assert all(isinstance(i, int) for i in self.data.keys()), "Make sure example_ids elements of the test data are of type int"

    def infere_summarization_model(self):
        """Function that inferes your model on a dictionary containing the input in {0:summary0, 1:summary1 ...} dict formate as used in previous round submission
    
            Uses and updates the following class attributes:
            Uses:
                self.data: A dictionary containing the input paragraph to summarize
                in the following formate:
                    {0:summary0, 
                     1:summary1 ...}
                self.model_path: path to the model used to generate the summaries
                self.device: to choose whether inference is on CPU or GPU
            Updates:
                self.results: A dictionary containing the results of the summarization 
                in the same formate as the input data

        """
        #################### START OF MODIFICATIONS ##################################
        self.results=infere_model(self.model_path,self.data,self.device)

        #################### END OF MODIFICATIONS ####################################
        #assertions
        assert all(isinstance(i, int) for i in self.results.keys()), "Make sure example_ids elements (key of self.results) are of type int"
        assert all(isinstance(i, str) for i in self.results.values()), "Make sure summary elements (value of self.results) are of type str"
        
        diff_sub = set(self.results.keys()) - set(self.data.keys())
        diff_base = set(self.data.keys()) - set(self.results.keys())
        
        assert len(diff_sub) == 0, f"Keys {diff_sub} is in submission but not in the input data keys"
        assert len(diff_base) == 0, f"Keys {diff_base} is in input data keys but not in submission"
    
    def save_results(self):
        save_file = os.path.join(self.result_save_path,self.leader_codalab_email)
        output_data=[{"example_id":i,"summary":self.results[i]} for i in sorted(self.results.keys())]
        with open(save_file, "w") as f:
            for json_obj in output_data:
                json.dump(json_obj, f,ensure_ascii=False)
                f.write("\n")
    def run_inference(self):
        print("loading the dataset")
        self.get_data()
        print('inferring the model')
        self.infere_summarization_model()
        print('Saving the results')
        self.save_results()

def parse_args():
    parser = argparse.ArgumentParser(
                    prog='AIC-1 Final Submission')
    
    parser.add_argument('-b', '--batch-size',default=1,type=int)      
    parser.add_argument('-m', '--model-dir',type=str)
    parser.add_argument('-l', '--leader-codalab-email',type=str)      
    parser.add_argument('-r', '--result-save-path',type=str)
    parser.add_argument('-v', '--val',action='store_true')
    parser.add_argument('-d', '--data-path',type=str)

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args=parse_args()
    sum_model=SummarizationModel(args)
    sum_model.run_inference()

        


