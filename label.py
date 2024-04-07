import os
import json
import tkinter as tk
import tkinter.messagebox
from PIL import Image, ImageTk

class ImageRotatorApp:
    def __init__(self, master, labels_path, output_path, all_image_index=500):
        
        self.config = {}
        
        self.output_path = output_path
        self.labels_path=labels_path
        self.master = master
        self.title = "双图旋转验证码标注系统"
        self.master.title(self.title)

        # Initialize variables
        self.angle = 0
        self.image_index = 1
        self.all_image_index = all_image_index

        if os.path.exists(f"{labels_path}/config.json"):
            with open(f"{labels_path}/config.json") as f:
                self.config = json.loads(f.read())
                keys = [int(k) for k in self.config.keys()]
                self.image_index = int(max(keys))+1

        # Load images
        self.image1 = Image.open(f"{labels_path}/bg_{self.image_index}.png")
        self.image2 = Image.open(f"{labels_path}/center_{self.image_index}.png")
        self.image1_tk = ImageTk.PhotoImage(self.image1)
        self.image2_tk = ImageTk.PhotoImage(self.image2)

        self.master.title(f"{self.title} - {self.image_index} / {self.all_image_index}")

        # Create canvas
        self.canvas = tk.Canvas(master, width=self.image1.width, height=self.image1.height)
        self.canvas.pack()

        # Display images on canvas
        self.image1_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image1_tk)
        self.center_x = (self.image1.width - self.image2.width) // 2
        self.center_y = (self.image1.height - self.image2.height) // 2
        self.image2_id = self.canvas.create_image(self.center_x, self.center_y, anchor=tk.NW, image=self.image2_tk)

        # Create rotation slider
        self.rotation_slider = tk.Scale(master, from_=0, to=360, orient=tk.HORIZONTAL, length=360, command=self.rotate_image)
        self.rotation_slider.pack()

        # Create angle entry
        self.input_frame = tk.Frame(master)
        self.input_frame.pack()

        self.decrease_button = tk.Button(self.input_frame, text="-", command=lambda: self.adjust_angle(-1))
        self.decrease_button.pack(side=tk.LEFT, pady=5)
        self.angle_var = tk.StringVar()
        self.angle_var.set(str(self.angle))
        validate_cmd = master.register(self.validate_input)
        self.angle_entry = tk.Entry(self.input_frame, textvariable=self.angle_var, validate="key", validatecommand=(validate_cmd, '%P'))
        self.angle_entry.pack(side=tk.LEFT)
        self.increase_button = tk.Button(self.input_frame, text="+", command=lambda: self.adjust_angle(1))
        self.increase_button.pack(side=tk.LEFT, pady=5)
        
        # Create "Next Group" button
        self.next_button = tk.Button(master, text="Next Group", command=self.next_group)
        self.next_button.pack(pady=10)
        
        # Create "Commit" button
        self.commit_button = tk.Button(master, text="Commit", command=self.commit_angle)
        self.commit_button.pack(pady=10)
        

    def rotate_image(self, angle):
        if not hasattr(self,"angle_entry"):
            return 
        self.angle = int(angle)
        self.angle_entry.delete(0, tk.END)
        self.angle_entry.insert(0, str(self.angle))

        # Rotate image2
        self.rotated_image2 = self.image2.rotate(-self.angle)
        self.rotated_image2_tk = ImageTk.PhotoImage(self.rotated_image2)
        self.canvas.itemconfig(self.image2_id, image=self.rotated_image2_tk)
        # Update image2 position to keep it centered
        self.canvas.coords(self.image2_id, self.center_x, self.center_y)
        
    def validate_input(self, event):
        value = self.angle_var.get()
        if not value.isdigit():
            self.angle_var.set(str(self.angle))
        else:
            self.rotation_slider.set(value)
            self.rotate_image(value)
            
    def adjust_angle(self, delta):
        new_angle = int(self.angle_var.get()) + delta
        if 0 <= new_angle <= 360:
            self.angle_var.set(str(new_angle))
            self.rotation_slider.set(new_angle)
            self.rotate_image(new_angle)
    
    def save_rotate_image(self, image_path, degrees_to_rotate, output_path):
        # 打开图像
        image = Image.open(image_path)
        
        # 旋转图像
        rotated_image = image.rotate(-degrees_to_rotate)
    
        # 保存旋转后的图像
        rotated_image.save(output_path)
    
        print("图像已旋转并保存到:", output_path)
    
    def commit_angle(self):
        angle = int(self.angle_var.get())
        self.config[str(self.image_index)] = angle
        with open(f"{self.labels_path}/config.json", "w") as f:
            f.write(json.dumps(self.config))
        self.save_rotate_image(f"{self.labels_path}/center_{self.image_index}.png", angle, f"{self.output_path}/center_{self.image_index}.png")
    def next_group(self):
        # Change images
        self.commit_angle()
        self.image_index += 1
        if not os.path.exists(f"{self.labels_path}/bg_{self.image_index}.png") or not os.path.exists(f"{self.labels_path}/center_{self.image_index}.png"):
            tk.messagebox.showinfo("提示", "文件未找到或者已经标注结束")
            return
        
        self.master.title(f"{self.title} - {self.image_index} / {self.all_image_index}")
        self.image1 = Image.open(f"{self.labels_path}/bg_{self.image_index}.png")
        self.image2 = Image.open(f"{self.labels_path}/center_{self.image_index}.png")

        self.image1_tk = ImageTk.PhotoImage(self.image1)
        self.image2_tk = ImageTk.PhotoImage(self.image2)

        # Update canvas
        self.canvas.itemconfig(self.image1_id, image=self.image1_tk)
        self.canvas.itemconfig(self.image2_id, image=self.image2_tk)
        self.rotation_slider.set(0)  # Reset slider
        self.angle = 0
        self.angle_entry.delete(0, tk.END)
        self.angle_entry.insert(0, str(self.angle))

def main():
    labels_path = "xhs_captcha_imgs" # 需要标注的图片路径
    output_path = "temp"    # 输出旋转到正确角度的验证码图片
    
    root = tk.Tk()
    app = ImageRotatorApp(root, labels_path, output_path)
    root.mainloop()

if __name__ == "__main__":
    main()
