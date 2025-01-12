# HFaceArchiver

#### INSTALLING
`pip install huggingface_hub`

INSTALL DOCS: https://huggingface.co/docs/huggingface_hub/en/installation

**DEFAULT BATCH SIZE: 2**

##### EXAMPLE
 **with default target**
`python main.py --repo <MODEL-NAME>`

 **with batch larger size**
`python main.py --repo <MODEL-NAME> --batch-size 2`
 
 **with download loc**
`python main.py --repo <MODEL-NAME> --target <DOWNLOAD_LOC>`
