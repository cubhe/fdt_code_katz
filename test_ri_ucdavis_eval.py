import argparse
import cv2
import os
import re
import numpy as np


def video_generate(folder_path,output_video_path):
    ri=np.load(folder_path+'/RI.npy')
    # ri_gt=np.load(folder_path+'/RI_gt.npy')
    frame_rate = 2  # 输出视频的帧率
    frame_size = (1000,1000)  # 输出视频的大小，请根据实际情况调整

    # 读取文件夹中的所有文件并按照数字索引排序
    file_list = os.listdir(folder_path)
    # 使用正则表达式提取文件名中的数字
    #file_list = sorted(file_list, key=lambda x: int(re.search(r'test_(\d+)_1.0.bmp', x).group(1)))

    # 创建OpenCV视频编写器对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)

    # 循环遍历每个文件，读取图像并添加到视频
    ri=np.abs(ri-1.33)*100
    ri_gt=(ri_gt-1.33)*100
    #ri =ri/np.max(ri)*255
    for i in range(ri.shape[2]):

        print(i)
        img=ri[:,:,i]
        img1=img/np.max(img)*255
        img=cv2.resize(img1,(800,800)).astype('uint8')
        img2=ri_gt[:,:,i]
        img2=img2/np.max(img2)*255
        img2=cv2.resize(img2,(800,800)).astype('uint8')
        #cv2.putText(img,f'RI_pred_{i}',(100,100),fontScale=1,fontFace=cv2.FONT_HERSHEY_SIMPLEX,color=(255,255,255),thickness=2)
        dif=np.abs(ri_gt[:,:,i]-ri[:,:,i])*100
        dif = cv2.resize(dif, (800, 800)).astype('uint8')
        cv2.imshow('test2', img2)
        cv2.imshow('test3', dif)
        cv2.imshow('test',img)
        #cv2.imwrite('pred_ri.bmp',img)
        #cv2.imwrite('gt_ri.png', img2)
        bar=np.zeros((255,255))
        for i in range(255):
            bar[i,:]=i
        #bar = cv2.applyColorMap(bar.astype('uint8'), cv2.COLORMAP_JET)
        #cv2.imwrite('bar.png', bar)
        cv2.waitKey(0)
        # 将帧写入视频
        video_writer.write(img)

    # 释放视频编写器和关闭所有OpenCV窗口
    video_writer.release()
    cv2.destroyAllWindows()

    print("视频合成完成！")
def video_generate2(folder_path_gt,folder_path_pred,output_video_path):
    ri=np.load(folder_path_pred+'/RI.npy')
    ri_gt=np.load(folder_path_gt+'/RI_gt.npy')
    frame_rate = 2  # 输出视频的帧率
    frame_size = (1000,1000)  # 输出视频的大小，请根据实际情况调整

    mse=np.mean(np.square(ri - ri_gt))
    print(mse)
    # 创建OpenCV视频编写器对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)

    # 循环遍历每个文件，读取图像并添加到视频
    ri=np.abs(ri-1.33)*100
    ri_gt=np.abs(ri_gt-1.33)*100
    #ri =ri/np.max(ri)*255
    for i in range(ri.shape[2]):

        print(i)
        img = ri[:, :, i]
        # img1=(img-np.min(ri))/(np.max(ri)-np.min(ri))*255
        img1 = (img - np.min(ri)) / (np.max(ri) - np.min(ri)) * 600
        img = cv2.resize(img1, (800, 800)).astype('uint8')
        img2 = ri_gt[:, :, i]
        img2 = (img2 - np.min(ri_gt)) / (np.max(ri_gt) - np.min(ri_gt)) * 600
        # img2 = (img2) / (np.max(ri_gt)) * 400
        img2 = cv2.resize(img2, (800, 800)).astype('uint8')
        # cv2.putText(img,f'RI_pred_{i}',(100,100),fontScale=1,fontFace=cv2.FONT_HERSHEY_SIMPLEX,color=(255,255,255),thickness=2)
        dif = np.abs(ri_gt[:, :, i] - ri[:, :, i]) * 100
        dif = cv2.resize(dif, (800, 800)).astype('uint8')
        # cv2.imshow('test2', img2)
        # cv2.imshow('test3', dif)
        # cv2.imshow('test',img)
        cv2.imwrite(f'./RI_pred/pred_ri_{i}.bmp', img)
        cv2.imwrite(f'./RI_pred/gt_ri_{i}.png', img2)
        img3 = np.hstack((img2, img))
        img3_color = cv2.applyColorMap(img3, cv2.COLORMAP_JET)
        cv2.putText(img3_color, f'simulated GT RI.  No:{i}', (100, 100), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    color=(255, 255, 255), thickness=2)
        cv2.putText(img3_color, 'pred by mlp', (800, 100), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    color=(255, 255, 255), thickness=2)
        cv2.imwrite(f'./RI_pred/comp_ri_{i}.png', img3)
        bar = np.zeros((255, 255))
        for i in range(255):
            bar[i, :] = i
        # bar = cv2.applyColorMap(bar.astype('uint8'), cv2.COLORMAP_JET)
        # cv2.imwrite('bar.png', bar)
        cv2.waitKey(0)
        # 将帧写入视频
        video_writer.write(img3_color)
        video_writer.write(img3_color)
        video_writer.write(img3_color)

    # 释放视频编写器和关闭所有OpenCV窗口
    video_writer.release()
    cv2.destroyAllWindows()

    print("视频合成完成！")
def video_generate_RI_real(ri, output_video_path):
    # ri = np.load(folder_path_pred + '/RI(1).npy')[:,:,:]
    # ri_gt = np.load(folder_path_gt + '/RI_gt.npy')
    frame_rate = 2  # 输出视频的帧率
    frame_size = (1000, 1000)  # 输出视频的大小，请根据实际情况调整

    #mse = np.mean(np.square(ri - ri_gt))
    #print(mse)
    # 创建OpenCV视频编写器对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)
    video_writer = cv2.VideoWriter(output_video_path  , fourcc, frame_rate, frame_size)
    # 循环遍历每个文件，读取图像并添加到视频
    # ri = np.abs(ri) * 1
    ri = np.abs(ri) / 0.05 * 255
    # ri_gt = np.abs(ri_gt - 1.33) * 100
    # ri =ri/np.max(ri)*255
    factor=255
    for i in range(ri.shape[1]):
        print(i, end=' ')
        id=4
        ref = ri[:, :, id]
        ref = (ref - np.min(ri)) / (np.max(ri) - np.min(ri)) * factor
        ref=ref.astype('uint8')
        ref = cv2.applyColorMap(ref.astype('uint8'), cv2.COLORMAP_JET)
        cv2.imshow('ref', ref)
        cv2.imwrite(f'./ri_mdck/ri_xy_{id}.bmp', ref)

        #import cv2
        from matplotlib import pyplot as plt
        # 读取图片
        image = ref  # 替换为你的图片路径
        # 将BGR图片转换为RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 显示图片的函数
        def show_image_with_click_event(image_rgb):
            # 创建一个显示图片的窗口
            fig, ax = plt.subplots(2,2)
            plt.subplots_adjust(wspace=0, hspace=0)
            ax[0][0].spines['top'].set_linewidth(0.9)
            ax[0][0].spines['bottom'].set_linewidth(0.1)
            ax[0][0].spines['right'].set_linewidth(0.9)
            ax[0][0].spines['left'].set_linewidth(0.1)
            ax[1][0].spines['top'].set_linewidth(0.9)
            ax[1][0].spines['bottom'].set_linewidth(0.1)
            ax[1][0].spines['right'].set_linewidth(0.9)
            ax[1][0].spines['left'].set_linewidth(0.1)
            ax[0][1].spines['top'].set_linewidth(0.9)
            ax[0][1].spines['bottom'].set_linewidth(0.1)
            ax[0][1].spines['right'].set_linewidth(0.9)
            ax[0][1].spines['left'].set_linewidth(0.1)
            ax[1][1].spines['top'].set_linewidth(0.9)
            ax[1][1].spines['bottom'].set_linewidth(0.1)
            ax[1][1].spines['right'].set_linewidth(0.9)
            ax[1][1].spines['left'].set_linewidth(0.1)
            # 显示图片
            ax[0][0].imshow(image_rgb)
            ax[0][0].axis('off')
            # 处理鼠标点击事件的函数
            dd=8
            def onclick(event):
                # 获取点击的坐标，注意坐标需要转换为整数
                ix, iy = int(event.xdata), int(event.ydata)
                # 获取点击位置的像素值，注意坐标顺序是[y, x]因为matplotlib的坐标系和图片矩阵的坐标系不同
                pixel_value = image[iy, ix, :]
                print(f"Clicked at ({ix}, {iy}) - Pixel Value: {pixel_value}")
                image_rgb_new=image_rgb.copy()

                image_rgb_new[0:dd, ix - 1:ix + 1] = 255
                image_rgb_new[-dd:, ix - 1:ix + 1] = 255
                image_rgb_new[iy - 1:iy + 1, 0:dd] = 255
                image_rgb_new[iy - 1:iy + 1, -dd:] = 255

                img = ri[iy, :, :]
                img = np.rot90(img)  # -133
                img1 = (img - np.min(ri)) / (np.max(ri) - np.min(ri)) * factor


                img1 = cv2.resize(img1, (1024, int(12*5/0.31))).astype('uint8')
                img1_color = cv2.applyColorMap(img1, cv2.COLORMAP_JET)


                img2 = ri[:, ix, :]
                img2 = (img2 - np.min(ri)) / (np.max(ri) - np.min(ri)) * factor

                img2 = cv2.resize(img2, (int(12*5/0.31), 1024)).astype('uint8')
                img2_color = cv2.applyColorMap(img2, cv2.COLORMAP_JET)


                cv2.imshow('test', img1_color)
                cv2.imshow('test2', img2_color)
                #save
                cv2.imwrite(f'./ri_mdck/ri_x_{ix}.bmp', img1_color)
                cv2.imwrite(f'./ri_mdck/ri_y_{iy}.bmp', img2_color)
                img1_color = cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB)
                img2_color = cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB)
                # ax.imshow(image_rgb)
                # return image_rgb
                image_rgb_new[iy - 1:iy + 1, :] = 250
                image_rgb_new[:, ix - 1:ix + 1] = 0
                img2_color[iy - 1:iy + 1, :] = 255
                img1_color[:, ix - 1:ix + 1] = 0
                ax[0][0].clear()
                ax[0][0].imshow(image_rgb_new)
                ax[1][0].clear()
                ax[1][0].imshow(img1_color)
                ax[0][1].clear()
                ax[0][1].imshow(img2_color)

                ax[0][0].axis('off')
                ax[1][0].axis('off')
                ax[1][1].axis('off')
                ax[0][1].axis('off')
                plt.draw()
            # 绑定点击事件
            fig.canvas.mpl_connect('button_press_event', onclick)
            #ax.imshow(a)
            # 显示窗口
            plt.show()
        # 调用函数显示图片
        show_image_with_click_event(image_rgb)



        img = ri[i, :, :]
        img=np.rot90(img)# -133
        # img1=(img-np.min(ri))/(np.max(ri)-np.min(ri))*255
        H = img.shape[0]
        img1 = (img - np.min(ri)) / (np.max(ri) - np.min(ri)) * 250
        img1 = cv2.resize(img1,  (1024,200)).astype('uint8')
        img1_color = cv2.applyColorMap(img1, cv2.COLORMAP_JET)

        img2 = ri[:, i, :]
        #img2 = np.rot90(img2)  # -133
        # img1=(img-np.min(ri))/(np.max(ri)-np.min(ri))*255
        H = img.shape[0]
        img2 = (img2 - np.min(ri)) / (np.max(ri) - np.min(ri)) * 250
        img2 = cv2.resize(img2, (200, 1024)).astype('uint8')
        #img1 = np.rot90(img1)
        img2_color = cv2.applyColorMap(img2, cv2.COLORMAP_JET)

        # cv2.putText(img1_color, f'pred by mlp{i}', (100, 100), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,color=(255, 255, 255), thickness=2)
        # cv2.putText(img2_color, f'pred by mlp{i}', (100, 100), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,color=(255, 255, 255), thickness=2)
        cv2.imshow('ref', ref.astype('uint8'))

        cv2.imshow('test', img1_color)
        cv2.imshow('test2', img2_color)

        cv2.waitKey(0)

def apply_custom_colormap(gray_image, palette):
    # Normalize the grayscale image to have values between 0 and len(palette)-1
    normalized_gray = cv2.normalize(gray_image, None, 0, len(palette) - 1, cv2.NORM_MINMAX)

    # Create an empty color image
    colored_image = np.zeros((*gray_image.shape, 3), dtype=np.uint8)

    # Apply the palette
    for i in range(len(palette)):
        colored_image[normalized_gray == i] = palette[i]

    return colored_image

def normalize_to_uint8(img, denom):
    if denom is None or denom <= 0:
        return np.zeros_like(img, dtype=np.uint8)
    img = np.clip(img / denom, 0.0, 1.0)
    return np.round(img * 255.0).astype(np.uint8)


def export_ucdavis_eval(
    ri_path,
    save_dir,
    colormap_path='colormap0627.npy',
    crop=80,
    ri_format='delta',
    norm='slice',
    output_prefix='ri',
):
    palette = np.load(colormap_path)
    palette = palette[:, [2, 1, 0]]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    ri = np.load(ri_path)[:, :, :]
    ri = ri.copy()

    if ri_format == 'abs':
        ri = ri - 1.33
    elif ri_format != 'delta':
        raise ValueError(f'Unknown ri_format: {ri_format}')

    ri[ri < 0] = 0

    if crop > 0:
        ri = ri[crop:-crop, crop:-crop, :]

    global_max = float(np.max(ri))
    stats_path = os.path.join(save_dir, f'{output_prefix}_stats.txt')
    with open(stats_path, 'w') as f:
        f.write(f'ri_path={ri_path}\n')
        f.write(f'ri_format={ri_format}\n')
        f.write(f'norm={norm}\n')
        f.write(f'crop={crop}\n')
        f.write(f'global_max={global_max:.8f}\n')
        for i in range(ri.shape[2]):
            img = ri[:, :, i]
            f.write(
                f'slice={i} min={float(np.min(img)):.8f} max={float(np.max(img)):.8f} '
                f'mean={float(np.mean(img)):.8f} std={float(np.std(img)):.8f}\n'
            )

    for i in range(ri.shape[2]):
        print(i)
        img = ri[:, :, i]
        if norm == 'slice':
            img_u8 = normalize_to_uint8(img, float(np.max(img)))
        elif norm == 'global':
            img_u8 = normalize_to_uint8(img, global_max)
        else:
            raise ValueError(f'Unknown norm: {norm}')
        img_color = apply_custom_colormap(img_u8, palette)
        cv2.imwrite(os.path.join(save_dir, f'{output_prefix}_{i}.bmp'), img_color)


def parse_args():
    parser = argparse.ArgumentParser(description='Export UCDavis eval colormap images for an RI volume.')
    parser.add_argument('--ri-path', default='./dataset/ucdavis_dx0.33/ucdavis.npy')
    parser.add_argument('--save-dir', default='./dataset/ucdavis_dx0.33')
    parser.add_argument('--colormap-path', default='colormap0627.npy')
    parser.add_argument('--crop', type=int, default=80)
    parser.add_argument('--ri-format', choices=['delta', 'abs'], default='delta')
    parser.add_argument('--norm', choices=['slice', 'global'], default='slice')
    parser.add_argument('--output-prefix', default='ri')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    export_ucdavis_eval(
        ri_path=args.ri_path,
        save_dir=args.save_dir,
        colormap_path=args.colormap_path,
        crop=args.crop,
        ri_format=args.ri_format,
        norm=args.norm,
        output_prefix=args.output_prefix,
    )
