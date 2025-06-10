#!/bin/bash
###eps4, seed 0
##M=16
#full shots
#cscFalse
python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/dtd/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "DescribableTextures" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/dtd/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/ucf101/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "UCF101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/ucf101/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_flowers/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordFlowers" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_flowers/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_pets/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_pets/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/caltech101/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "Caltech101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/caltech101/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/fgvc_aircraft/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "FGVCAircraft" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/fgvc_aircraft/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/eurosat/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "EuroSAT" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/eurosat/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1
	
#cscTrue
python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/dtd/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "DescribableTextures" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/dtd/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/ucf101/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "UCF101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/ucf101/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_flowers/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordFlowers" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_flowers/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_pets/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_pets/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/caltech101/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "Caltech101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/caltech101/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/fgvc_aircraft/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "FGVCAircraft" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/fgvc_aircraft/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/eurosat/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "EuroSAT" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/eurosat/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1
	
#16 shots
#end
#cscFalse
python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/dtd/APT/vit_b32_16shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "DescribableTextures" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/dtd/APT/vit_b32_16shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/ucf101/APT/vit_b32_16shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "UCF101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/ucf101/APT/vit_b32_16shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_flowers/APT/vit_b32_16shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordFlowers" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_flowers/APT/vit_b32_16shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_pets/APT/vit_b32_16shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_pets/APT/vit_b32_16shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 2

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/caltech101/APT/vit_b32_16shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "Caltech101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/caltech101/APT/vit_b32_16shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/fgvc_aircraft/APT/vit_b32_16shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "FGVCAircraft" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/fgvc_aircraft/APT/vit_b32_16shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/eurosat/APT/vit_b32_16shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "EuroSAT" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/eurosat/APT/vit_b32_16shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1
	
#cscTrue
python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/dtd/APT/vit_b32_16shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "DescribableTextures" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/dtd/APT/vit_b32_16shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/ucf101/APT/vit_b32_16shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "UCF101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/ucf101/APT/vit_b32_16shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_flowers/APT/vit_b32_16shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordFlowers" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_flowers/APT/vit_b32_16shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_pets/APT/vit_b32_16shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_pets/APT/vit_b32_16shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/caltech101/APT/vit_b32_16shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "Caltech101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/caltech101/APT/vit_b32_16shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/fgvc_aircraft/APT/vit_b32_16shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "FGVCAircraft" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/fgvc_aircraft/APT/vit_b32_16shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/eurosat/APT/vit_b32_16shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "EuroSAT" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/eurosat/APT/vit_b32_16shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

#front
#cscFalse
 python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/dtd/APT/vit_b32_16shots/nctx16_cscFalse_ctpfront/eps4_alpha2.67_step3/seed0" \
    --dataset "DescribableTextures" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/dtd/APT/vit_b32_16shots/nctx16_cscFalse_ctpfront/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/ucf101/APT/vit_b32_16shots/nctx16_cscFalse_ctpfront/eps4_alpha2.67_step3/seed0" \
    --dataset "UCF101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/ucf101/APT/vit_b32_16shots/nctx16_cscFalse_ctpfront/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_flowers/APT/vit_b32_16shots/nctx16_cscFalse_ctpfront/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordFlowers" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_flowers/APT/vit_b32_16shots/nctx16_cscFalse_ctpfront/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_pets/APT/vit_b32_16shots/nctx16_cscFalse_ctpfront/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_pets/APT/vit_b32_16shots/nctx16_cscFalse_ctpfront/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/caltech101/APT/vit_b32_16shots/nctx16_cscFalse_ctpfront/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/caltech101/APT/vit_b32_16shots/nctx16_cscFalse_ctpfront/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/fgvc_aircraft/APT/vit_b32_16shots/nctx16_cscFalse_ctpfront/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/fgvc_aircraft/APT/vit_b32_16shots/nctx16_cscFalse_ctpfront/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/eurosat/APT/vit_b32_16shots/nctx16_cscFalse_ctpfront/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/eurosat/APT/vit_b32_16shots/nctx16_cscFalse_ctpfront/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

#cscTrue
python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/dtd/APT/vit_b32_16shots/nctx16_cscTrue_ctpfront/eps4_alpha2.67_step3/seed0" \
    --dataset "DescribableTextures" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/dtd/APT/vit_b32_16shots/nctx16_cscTrue_ctpfront/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/ucf101/APT/vit_b32_16shots/nctx16_cscTrue_ctpfront/eps4_alpha2.67_step3/seed0" \
    --dataset "UCF101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/ucf101/APT/vit_b32_16shots/nctx16_cscTrue_ctpfront/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_flowers/APT/vit_b32_16shots/nctx16_cscTrue_ctpfront/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordFlowers" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_flowers/APT/vit_b32_16shots/nctx16_cscTrue_ctpfront/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_pets/APT/vit_b32_16shots/nctx16_cscTrue_ctpfront/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_pets/APT/vit_b32_16shots/nctx16_cscTrue_ctpfront/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1
	
python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/caltech101/APT/vit_b32_16shots/nctx16_cscTrue_ctpfront/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/caltech101/APT/vit_b32_16shots/nctx16_cscTrue_ctpfront/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/fgvc_aircraft/APT/vit_b32_16shots/nctx16_cscTrue_ctpfront/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/fgvc_aircraft/APT/vit_b32_16shots/nctx16_cscTrue_ctpfront/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/eurosat/APT/vit_b32_16shots/nctx16_cscTrue_ctpfront/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/eurosat/APT/vit_b32_16shots/nctx16_cscTrue_ctpfront/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1
	
#middle
#cscFalse
 python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/dtd/APT/vit_b32_16shots/nctx16_cscFalse_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --dataset "DescribableTextures" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/dtd/APT/vit_b32_16shots/nctx16_cscFalse_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/ucf101/APT/vit_b32_16shots/nctx16_cscFalse_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --dataset "UCF101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/ucf101/APT/vit_b32_16shots/nctx16_cscFalse_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_flowers/APT/vit_b32_16shots/nctx16_cscFalse_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordFlowers" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_flowers/APT/vit_b32_16shots/nctx16_cscFalse_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_pets/APT/vit_b32_16shots/nctx16_cscFalse_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_pets/APT/vit_b32_16shots/nctx16_cscFalse_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/caltech101/APT/vit_b32_16shots/nctx16_cscFalse_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/caltech101/APT/vit_b32_16shots/nctx16_cscFalse_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/fgvc_aircraft/APT/vit_b32_16shots/nctx16_cscFalse_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/fgvc_aircraft/APT/vit_b32_16shots/nctx16_cscFalse_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/eurosat/APT/vit_b32_16shots/nctx16_cscFalse_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/eurosat/APT/vit_b32_16shots/nctx16_cscFalse_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

#cscTrue
python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/dtd/APT/vit_b32_16shots/nctx16_cscTrue_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --dataset "DescribableTextures" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/dtd/APT/vit_b32_16shots/nctx16_cscTrue_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/ucf101/APT/vit_b32_16shots/nctx16_cscTrue_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --dataset "UCF101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/ucf101/APT/vit_b32_16shots/nctx16_cscTrue_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_flowers/APT/vit_b32_16shots/nctx16_cscTrue_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordFlowers" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_flowers/APT/vit_b32_16shots/nctx16_cscTrue_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_pets/APT/vit_b32_16shots/nctx16_cscTrue_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_pets/APT/vit_b32_16shots/nctx16_cscTrue_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1
	
python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/caltech101/APT/vit_b32_16shots/nctx16_cscTrue_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/caltech101/APT/vit_b32_16shots/nctx16_cscTrue_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/fgvc_aircraft/APT/vit_b32_16shots/nctx16_cscTrue_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/fgvc_aircraft/APT/vit_b32_16shots/nctx16_cscTrue_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/eurosat/APT/vit_b32_16shots/nctx16_cscTrue_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/eurosat/APT/vit_b32_16shots/nctx16_cscTrue_ctpmiddle/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1
	
#4 shots
#cscFalse
python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/dtd/APT/vit_b32_ep100_4shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "DescribableTextures" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/dtd/APT/vit_b32_ep100_4shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/ucf101/APT/vit_b32_ep100_4shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "UCF101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/ucf101/APT/vit_b32_ep100_4shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_flowers/APT/vit_b32_ep100_4shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordFlowers" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_flowers/APT/vit_b32_ep100_4shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_pets/APT/vit_b32_ep100_4shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_pets/APT/vit_b32_ep100_4shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 2

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/caltech101/APT/vit_b32_ep100_4shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "Caltech101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/caltech101/APT/vit_b32_ep100_4shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/fgvc_aircraft/APT/vit_b32_ep100_4shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "FGVCAircraft" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/fgvc_aircraft/APT/vit_b32_ep100_4shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/eurosat/APT/vit_b32_ep100_4shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "EuroSAT" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/eurosat/APT/vit_b32_ep100_4shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1
	
#cscTrue
python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/dtd/APT/vit_b32_ep100_4shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "DescribableTextures" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/dtd/APT/vit_b32_ep100_4shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/ucf101/APT/vit_b32_ep100_4shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "UCF101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/ucf101/APT/vit_b32_ep100_4shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_flowers/APT/vit_b32_ep100_4shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordFlowers" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_flowers/APT/vit_b32_ep100_4shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_pets/APT/vit_b32_ep100_4shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_pets/APT/vit_b32_ep100_4shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/caltech101/APT/vit_b32_ep100_4shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "Caltech101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/caltech101/APT/vit_b32_ep100_4shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/fgvc_aircraft/APT/vit_b32_ep100_4shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "FGVCAircraft" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/fgvc_aircraft/APT/vit_b32_ep100_4shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/eurosat/APT/vit_b32_ep100_4shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "EuroSAT" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/eurosat/APT/vit_b32_ep100_4shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

#1shots
#cscFalse
python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/dtd/APT/vit_b32_ep50_1shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "DescribableTextures" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/dtd/APT/vit_b32_ep50_1shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/ucf101/APT/vit_b32_ep50_1shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "UCF101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/ucf101/APT/vit_b32_ep50_1shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_flowers/APT/vit_b32_ep50_1shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordFlowers" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_flowers/APT/vit_b32_ep50_1shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_pets/APT/vit_b32_ep50_1shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_pets/APT/vit_b32_ep50_1shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 2

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/caltech101/APT/vit_b32_ep50_1shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "Caltech101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/caltech101/APT/vit_b32_ep50_1shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/fgvc_aircraft/APT/vit_b32_ep50_1shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "FGVCAircraft" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/fgvc_aircraft/APT/vit_b32_ep50_1shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/eurosat/APT/vit_b32_ep50_1shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "EuroSAT" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/eurosat/APT/vit_b32_ep50_1shots/nctx16_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1
	
#cscTrue
python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/dtd/APT/vit_b32_ep50_1shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "DescribableTextures" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/dtd/APT/vit_b32_ep50_1shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/ucf101/APT/vit_b32_ep50_1shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "UCF101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/ucf101/APT/vit_b32_ep50_1shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_flowers/APT/vit_b32_ep50_1shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordFlowers" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_flowers/APT/vit_b32_ep50_1shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_pets/APT/vit_b32_ep50_1shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_pets/APT/vit_b32_ep50_1shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/caltech101/APT/vit_b32_ep50_1shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "Caltech101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/caltech101/APT/vit_b32_ep50_1shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/fgvc_aircraft/APT/vit_b32_ep50_1shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "FGVCAircraft" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/fgvc_aircraft/APT/vit_b32_ep50_1shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/eurosat/APT/vit_b32_ep50_1shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "EuroSAT" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/eurosat/APT/vit_b32_ep50_1shots/nctx16_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

#M=1
#full shots
#cscFalse
python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/dtd/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "DescribableTextures" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/dtd/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/ucf101/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "UCF101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/ucf101/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_flowers/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordFlowers" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_flowers/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_pets/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_pets/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/caltech101/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "Caltech101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/caltech101/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/fgvc_aircraft/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "FGVCAircraft" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/fgvc_aircraft/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/eurosat/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "EuroSAT" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/eurosat/APT/vit_b32_-1shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1
	
#cscTrue
python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/dtd/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "DescribableTextures" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/dtd/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/ucf101/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "UCF101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/ucf101/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_flowers/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordFlowers" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_flowers/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_pets/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_pets/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/caltech101/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "Caltech101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/caltech101/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/fgvc_aircraft/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "FGVCAircraft" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/fgvc_aircraft/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/eurosat/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "EuroSAT" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/eurosat/APT/vit_b32_-1shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1
	
#16 shots
#cscFalse
python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/dtd/APT/vit_b32_16shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "DescribableTextures" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/dtd/APT/vit_b32_16shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/ucf101/APT/vit_b32_16shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "UCF101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/ucf101/APT/vit_b32_16shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_flowers/APT/vit_b32_16shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordFlowers" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_flowers/APT/vit_b32_16shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_pets/APT/vit_b32_16shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_pets/APT/vit_b32_16shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 2

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/caltech101/APT/vit_b32_16shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "Caltech101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/caltech101/APT/vit_b32_16shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/fgvc_aircraft/APT/vit_b32_16shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "FGVCAircraft" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/fgvc_aircraft/APT/vit_b32_16shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/eurosat/APT/vit_b32_16shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "EuroSAT" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/eurosat/APT/vit_b32_16shots/nctx1_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1
	
#cscTrue
python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/dtd/APT/vit_b32_16shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "DescribableTextures" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/dtd/APT/vit_b32_16shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/ucf101/APT/vit_b32_16shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "UCF101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/ucf101/APT/vit_b32_16shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_flowers/APT/vit_b32_16shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordFlowers" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_flowers/APT/vit_b32_16shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_pets/APT/vit_b32_16shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_pets/APT/vit_b32_16shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/caltech101/APT/vit_b32_16shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "Caltech101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/caltech101/APT/vit_b32_16shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/fgvc_aircraft/APT/vit_b32_16shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "FGVCAircraft" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/fgvc_aircraft/APT/vit_b32_16shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/eurosat/APT/vit_b32_16shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "EuroSAT" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/eurosat/APT/vit_b32_16shots/nctx1_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1
	
#M=4
#16 shots
#cscFalse
python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/dtd/APT/vit_b32_16shots/nctx4_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "DescribableTextures" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/dtd/APT/vit_b32_16shots/nctx4_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/ucf101/APT/vit_b32_16shots/nctx4_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "UCF101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/ucf101/APT/vit_b32_16shots/nctx4_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_flowers/APT/vit_b32_16shots/nctx4_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordFlowers" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_flowers/APT/vit_b32_16shots/nctx4_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_pets/APT/vit_b32_16shots/nctx4_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_pets/APT/vit_b32_16shots/nctx4_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 2

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/caltech101/APT/vit_b32_16shots/nctx4_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "Caltech101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/caltech101/APT/vit_b32_16shots/nctx4_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/fgvc_aircraft/APT/vit_b32_16shots/nctx4_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "FGVCAircraft" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/fgvc_aircraft/APT/vit_b32_16shots/nctx4_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/eurosat/APT/vit_b32_16shots/nctx4_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "EuroSAT" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/eurosat/APT/vit_b32_16shots/nctx4_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1
	
#cscTrue
python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/dtd/APT/vit_b32_16shots/nctx4_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "DescribableTextures" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/dtd/APT/vit_b32_16shots/nctx4_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/ucf101/APT/vit_b32_16shots/nctx4_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "UCF101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/ucf101/APT/vit_b32_16shots/nctx4_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_flowers/APT/vit_b32_16shots/nctx4_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordFlowers" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_flowers/APT/vit_b32_16shots/nctx4_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_pets/APT/vit_b32_16shots/nctx4_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_pets/APT/vit_b32_16shots/nctx4_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 2

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/caltech101/APT/vit_b32_16shots/nctx4_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "Caltech101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/caltech101/APT/vit_b32_16shots/nctx4_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/fgvc_aircraft/APT/vit_b32_16shots/nctx4_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "FGVCAircraft" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/fgvc_aircraft/APT/vit_b32_16shots/nctx4_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/eurosat/APT/vit_b32_16shots/nctx4_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "EuroSAT" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/eurosat/APT/vit_b32_16shots/nctx4_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

#M=8
#cscFalse
python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/dtd/APT/vit_b32_16shots/nctx8_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "DescribableTextures" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/dtd/APT/vit_b32_16shots/nctx8_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/ucf101/APT/vit_b32_16shots/nctx8_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "UCF101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/ucf101/APT/vit_b32_16shots/nctx8_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_flowers/APT/vit_b32_16shots/nctx8_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordFlowers" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_flowers/APT/vit_b32_16shots/nctx8_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_pets/APT/vit_b32_16shots/nctx8_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_pets/APT/vit_b32_16shots/nctx8_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/caltech101/APT/vit_b32_16shots/nctx8_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "Caltech101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/caltech101/APT/vit_b32_16shots/nctx8_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/fgvc_aircraft/APT/vit_b32_16shots/nctx8_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "FGVCAircraft" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/fgvc_aircraft/APT/vit_b32_16shots/nctx8_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/eurosat/APT/vit_b32_16shots/nctx8_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "EuroSAT" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/eurosat/APT/vit_b32_16shots/nctx8_cscFalse_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1
	
#cscTrue
python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/dtd/APT/vit_b32_16shots/nctx8_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "DescribableTextures" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/dtd/APT/vit_b32_16shots/nctx8_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/ucf101/APT/vit_b32_16shots/nctx8_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "UCF101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/ucf101/APT/vit_b32_16shots/nctx8_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_flowers/APT/vit_b32_16shots/nctx8_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordFlowers" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_flowers/APT/vit_b32_16shots/nctx8_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/oxford_pets/APT/vit_b32_16shots/nctx8_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "OxfordPets" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/oxford_pets/APT/vit_b32_16shots/nctx8_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/caltech101/APT/vit_b32_16shots/nctx8_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "Caltech101" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/caltech101/APT/vit_b32_16shots/nctx8_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/fgvc_aircraft/APT/vit_b32_16shots/nctx8_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "FGVCAircraft" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/fgvc_aircraft/APT/vit_b32_16shots/nctx8_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1

python evaluate.py "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output/eurosat/APT/vit_b32_16shots/nctx8_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --dataset "EuroSAT" \
    --atk-prompt "'a photo of a {}'" \
    --attack "pgd" \
    --save-img \
    --save-path "/home/khoahocmaytinh2022/Desktop/MinhNhut/Dassl.pytorch/APT/apt/output_orig/eurosat/APT/vit_b32_16shots/nctx8_cscTrue_ctpend/eps4_alpha2.67_step3/seed0" \
    --num-imgs 10 \
    --seed 42 \
    --topk 1
