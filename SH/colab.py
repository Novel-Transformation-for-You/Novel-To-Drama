!git clone https://github.com/YueChenkkk/CSN-SAPR.git

%cd CSN-SAPR

!chmod +x run_train.sh

!pip install fastprogress==1.0.0
!pip install python==3.6
!pip install jieba==0.42.1
!pip install numpy==1.19.1
!pip install torch==1.2.0
!pip install tensorflow==2.0.0
!pip install transformers==4.6.1

!sudo apt-get update
!sudo apt-get install python3.6

!sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6

ROOT_DIR=""
BERT_PRETRAINED_DIR="beomi/kcbert-large"
CHECKPOINT_DIR=""
DATA_PREFIX="./data"

!source ${ROOT_DIR}/.bashrc

!CUDA_VISIBLE_DEVICES=0 python train.py \
--model_name CSN \
--pooling_type max_pooling \
--dropout 0.5 \
--optimizer adam \
--margin 1.0 \
--lr 2e-5 \
--num_epochs 50 \
--batch_size 16 \
--patience 10 \
--bert_pretrained_dir beomi/kcbert-large \
--train_file \
/content/CSN-SAPR/data/train/train_unsplit.txt \
--dev_file \
/content/CSN-SAPR/data/dev/dev_unsplit.txt \
--test_file \
/content/CSN-SAPR/data/test/test_unsplit.txt \
--name_list_path \
/content/CSN-SAPR/data/name_list.txt \
--length_limit 510 \
--checkpoint_dir ""
