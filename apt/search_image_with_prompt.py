import os
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from clip import model as clip_model  # Import module model từ clip
from clip import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

def load_image_dataset(image_dir):
    """Tải tất cả ảnh từ thư mục `image_dir` và các thư mục con, lấy tên lớp từ tên thư mục con."""
    images = []
    for root, _, files in os.walk(image_dir):
        class_name = os.path.basename(root)  # Lấy tên thư mục con làm tên lớp
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)
                image = Image.open(img_path).convert("RGB")
                images.append((image, filename, class_name))
    return images

def preprocess_images(images, preprocess):
    """Tiền xử lý ảnh theo yêu cầu của CLIP."""
    return [(preprocess(image), filename) for image, filename in images]

def load_clip_model(backbone, device):
    """Load mô hình CLIP từ tên hoặc từ tệp mô hình .pth.tar."""
    if os.path.isfile(backbone):
        # Trường hợp backbone là một tệp .pth.tar
        print(f"Loading custom CLIP model from file: {backbone}")
        state_dict = torch.load(backbone, map_location=device)
        model = clip_model.build_model(state_dict)  # Sử dụng hàm build_model từ clip.model
        input_resolution = model.visual.input_resolution if hasattr(model.visual, 'input_resolution') else 224
        preprocess = clip._transform(input_resolution)  # Sử dụng hàm tiền xử lý mặc định của clip
    else:
        # Trường hợp backbone là tên của mô hình có sẵn
        print(f"Loading pretrained CLIP model: {backbone}")
        model, preprocess = clip.load(backbone, device=device)
    
    model = model.to(device).eval()
    return model, preprocess

def find_top_k_images(prompt, images, model, device, k):
    """Tìm top-k ảnh gần nhất với câu prompt dựa trên embedding CLIP."""
    with torch.no_grad():
        # Tạo embedding cho câu prompt
        prompt_token = clip.tokenize([prompt]).to(device)
        prompt_features = model.encode_text(prompt_token)

        # Tạo embedding cho tất cả ảnh
        image_tensors = torch.stack([image_tensor for image_tensor, _ in images]).to(device)
        image_features = model.encode_image(image_tensors)

        # Tính khoảng cách cosine giữa prompt và ảnh
        similarity = (prompt_features @ image_features.T).squeeze()
        top_k_indices = similarity.topk(k).indices.tolist()  # Lấy chỉ số của top-k ảnh

        # Trả về top-k ảnh và tên file tương ứng
        top_k_images = [(images[idx][0], images[idx][1], similarity[idx].item()) for idx in top_k_indices]
        return top_k_images

def display_images_with_prompt(images, prompt):
    """Hiển thị n ảnh với câu prompt và tên file."""
    n = len(images)
    plt.figure(figsize=(15, 5))
    for i, (image, filename, similarity) in enumerate(images):
        plt.subplot(1, n, i + 1)
        plt.imshow(image.permute(1, 2, 0))  # Đảo trục để hiển thị ảnh
        plt.title(f"{filename}\nSimilarity: {similarity:.2f}")
        plt.axis("off")
    plt.suptitle(f"Prompt: {prompt}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Find top-k images matching a prompt using CLIP.")
    parser.add_argument("--image-dir", type=str, required=True, help="Thư mục chứa ảnh cần tìm.")
    parser.add_argument("--prompt", type=str, required=True, help="Câu prompt dùng để tìm kiếm ảnh.")
    parser.add_argument("--backbone", type=str, required=True, help="Tên backbone của CLIP hoặc đường dẫn tới tệp .pth.tar")
    parser.add_argument("--top-k", type=int, default=1, help="Số lượng ảnh gần nhất cần hiển thị")
    args = parser.parse_args()

    # Thiết lập thiết bị
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tải mô hình CLIP và hàm tiền xử lý với backbone chỉ định
    model, preprocess = load_clip_model(args.backbone, device)

    # Tải và tiền xử lý ảnh
    images = load_image_dataset(args.image_dir)
    processed_images = preprocess_images(images, preprocess)

    # Tìm top-k ảnh gần nhất với prompt
    top_k_images = find_top_k_images(args.prompt, processed_images, model, device, args.top_k)

    # Hiển thị n ảnh gần nhất với câu prompt
    display_images_with_prompt(top_k_images, args.prompt)

if __name__ == "__main__":
    main()
