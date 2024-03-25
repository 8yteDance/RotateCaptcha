from PIL import Image
import os

def rotate_image(image_path, degrees_to_rotate, output_path):
    # 打开图像
    image = Image.open(image_path)
    
    print(degrees_to_rotate)
    # 旋转图像
    rotated_image = image.rotate(-degrees_to_rotate)

    # 保存旋转后的图像
    rotated_image.save(output_path)

    print("图像已旋转并保存到:", output_path)

def rotate_and_save_images(input_dir, output_dir, degree_step=1):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录下的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # 打开图像
            image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path)

            # 生成旋转的图像并保存
            for degree in range(0, 360, degree_step):
                rotated_image = image.rotate(degree)
                output_filename = f"{os.path.splitext(filename)[0]}_{degree}.png"
                output_path = os.path.join(output_dir, output_filename)
                rotated_image.save(output_path)
                print(f"已生成并保存旋转 {degree} 度的图像：{output_path}")




    
if __name__ == '__main__':
    
    # 输入目录和输出目录
    input_directory = "temp"
    output_directory = "360"
    
    # 调用函数生成旋转的图像并保存
    rotate_and_save_images(input_directory, output_directory)
    
