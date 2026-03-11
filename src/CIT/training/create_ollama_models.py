#this script aims at creating a modefile from a finetuned model folder (containing the adapter weights) and a base modelfile
import os
from argparse import ArgumentParser

from utils import check_has_subfolers, create_modelfile_from_base_model_file

parser= ArgumentParser()

parser.add_argument(
    "--base_modelfile_path",
    type=str,
    default="./src/CIT/models/modelfiles/llama3.1_base_modelfile",
    help="base modelfile path",
)
parser.add_argument(
    "--output_models_dir",
    type=str,
    default="./src/CIT/training/models/cv/ft_22.5",
    help="output directory of many models or one model",
)

parser.add_argument(
    "--output_modelfiles_dir",
    type=str,
    default="./src/CIT/training/models/cv/modelfiles/ft_22.5",
    help="output directory for the new modefiles",
)

if __name__ == "__main__":
    args = parser.parse_args()
    base_modelfile_path = args.base_modelfile_path
    output_models_dir = args.output_models_dir
    output_modelfiles_dir = args.output_modelfiles_dir
    if not os.path.exists(output_modelfiles_dir):
        os.makedirs(output_modelfiles_dir)

    #check if there are folders in the output_models_dir
    if not check_has_subfolers(output_models_dir):#only one model
        print(f"Only one model found in {output_models_dir}.")
        adapter_path=output_models_dir
        filename = os.path.basename(adapter_path)
        modelfile_path= os.path.join(output_modelfiles_dir, filename)
        create_modelfile_from_base_model_file(
            base_modelfile_path=base_modelfile_path,
            output_model_path=adapter_path,
            output_modelfile_path=modelfile_path
        )
        
    else:#create modelfile for each adapter
        for adapter_path in os.listdir(output_models_dir):
            filename = os.path.basename(adapter_path)
            modelfile_path= os.path.join(output_modelfiles_dir, filename)
            output_model_path=os.path.join(output_models_dir, adapter_path)
            create_modelfile_from_base_model_file(
                base_modelfile_path=base_modelfile_path,
                output_model_path=output_model_path,
                output_modelfile_path=modelfile_path
            )

        





