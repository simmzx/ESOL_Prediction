import argparse
import importlib
import os
import sys

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-gpu',action='store_true');args=parser.parse_args()
    required=('numpy','pandas','scipy','sklearn','rdkit','xgboost','torch','matplotlib','joblib')
    missing=[]
    for name in required:
        try:
            m=importlib.import_module(name)
            print(f'[ENV] {name}: {getattr(m,"__version__","installed")}',flush=True)
        except Exception as e:
            missing.append(name);print(f'[ENV] MISSING {name}: {e}',flush=True)
    if missing:sys.exit('Missing modules: '+', '.join(missing))
    import torch
    print('[ENV] Python',sys.version.split()[0],'CUDA_VISIBLE_DEVICES',os.getenv('CUDA_VISIBLE_DEVICES'))
    print('[ENV] torch CUDA built with:',torch.version.cuda,'CUDA available:',torch.cuda.is_available(),flush=True)
    if torch.cuda.is_available():
        print('[ENV] Visible accelerator:',torch.cuda.get_device_name(0))
    elif args.require_gpu:
        sys.exit('GPU 1 NOT usable in torch. Confirm server driver and install a CUDA-enabled PyTorch build; --allow-cpu for diagnostic runs.')

if __name__=='__main__':main()
