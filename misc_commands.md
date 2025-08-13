## Workflow ##

python test.py /path/to/logdir 0306_0001,0306_0002,0306_0003 /some/imgdir,/some/other/imgdir \
    --names "L3C,RGB Shared,RGB" --recursive=auto

L3C:

python l3c.py ../models 0524_0005 enc ../test/ILSVRC2012_val_00000003.JPEG out.l3c

python l3c.py ../models 0524_0005 dec ./out.l3c decoded.png


DO NOT MOUNT TO WINDOWS!!!!

source /opt/conda/etc/profile.d/conda.sh
conda activate l3c_env


export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH




1. Download the Miniconda installer

Run this command:

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

2. Run the installer script

bash Miniconda3-latest-Linux-x86_64.sh

    Follow the prompts.

    Accept the license.

    Choose install location (default is fine).

    When asked “Do you wish the installer to initialize Miniconda3 by running conda init?” say yes.

3. Restart your shell

Close your terminal and open it again, or run:

source ~/.bashrc

4. Verify installation

conda --version

You should now see something like:

conda 23.1.0

5. Then create your environment

conda create --name l3c_env python=3.7 pip --yes
conda activate l3c_env

If you get stuck or want me to help you step-by-step just say so! You’re very close.



SREC: 

Note I am using a 3060 so i cant use docker pull nvcr.io/nvidia/pytorch:19.06-py3 as described in the SReC paper -> FALSE WARNING APPARENTLY

USE pip install grpcio==1.42.0

List of useful commands I need to develop for my convenience:

docker run -it --gpus all --rm -v "C:\CodingProjects\Python\SREC:/SReC" nvcr.io/nvidia/pytorch:19.06-py3 bash


Compress in home directory SReC using: 

python3 -um src.encode \
  --path /SReC/data/validation/raw \
  --file /SReC/datasets/open_images_val.txt \
  --save-path /SReC/data/validation/compr \
  --load /SReC/models/openimages.pth \
  --decode


Decompress in home directory SReC using:
python3 -um src.decode \
  --path /SReC/data/validation/compr \
  --file /SReC/datasets/srec_file_list.txt \
  --save-path /SReC/data/validation/decompr \
  --load /SReC/models/openimages.pth

Evaluate:
python3 -um src.eval \
  --path /SReC/data/validation/raw \
  --file /SReC/datasets/open_images_val.txt \
  --load /SReC/models/openimages.pth



Worked in some container:
/SReCroot@061cb0f38a72:/SReC# cpython3 -um src.encode \
2025-06-06 15:40:11


>   --path /SReC/data/validation/raw \
2025-06-06 15:40:11


>   --file /SReC/datasets/test.txt \
2025-06-06 15:40:11


>   --save-path /SReC/data/validation/compr \
2025-06-06 15:40:12


>   --load /SReC/models/openimages.pth



python3 -um src.encode \
  --path /SReC/test_data \
  --file /SReC/datasets/test2.txt \
  --save-path /SReC/test_data_compr \
  --load /SReC/models/openimages.pth \
  --decode


Decompress in home directory SReC using:
python3 -um src.decode \
  --path /SReC/data/validation/compr \
  --file /SReC/datasets/srec_file_list.txt \
  --save-path /SReC/data/validation/decompr \
  --load /SReC/models/openimages.pth




if everything goes well:
python3 -um src.encode \
  --path ./data/raw \
  --file ./datasets/open_images_val.txt \
  --save-path ./data/compr \
  --load ./models/openimages.pth \
  --decode


python3 -um src.decode \
  --path ./data_cmpr \
  --file ./datasets/srec_names.txt \
  --save-path ./data_decompr \
  --load ./models/openimages.pth



