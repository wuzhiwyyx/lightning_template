'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-24 22:19:10
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-25 00:59:15
 # @ Description: Visualizing tools.
 '''


import io
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import numpy as np
from tqdm import tqdm
import cv2


def fig2npy(fig):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr

def draw_seq_res(data_item, pred):
    hor, box, cls = data_item
    seq_len, channels, height, width, nobj = *hor.shape, box.shape[0]
    imgs = hor[:, :2, :, :]
    imgs = torch.stack([torch.complex(x[0], x[1]).abs() for x in imgs])
    fig, axes = plt.subplots(1, seq_len, figsize=(48, 16))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.01,)
    box = box.permute(1, 0, 2).contiguous()
    pred = pred.permute(1, 0, 2).contiguous()
    for i, (rf, b, p) in enumerate(zip(imgs, box, pred)):
        ax = axes[i]
        for j, bb in enumerate(b):
            x, y, x_, y_ = bb
            rect = Rectangle((x, y), x_ - x, y_ - y, linewidth=2, edgecolor='w', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y-2, 'Person%d' % j, fontsize=18, color='w')
        for k, bb in enumerate(p):
            c, x, y, x_, y_ = bb
            rect = Rectangle((x, y), x_ - x, y_ - y, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y-2, 'Person%d\n%.4f' % (k, c), fontsize=18, color='r')
        ax.matshow(rf)
        ax.set_axis_off()

def visualize(dataset, preds, frame_idx=0):
    frames = []
    backend_ =  mpl.get_backend() 
    mpl.use("Agg")  # Prevent showing stuff
    pbar = tqdm(total=len(dataset), desc='Visualizing')
    for i, (d, pred) in enumerate(zip(dataset, preds)):
        # hor shape: seq_len, channels, height, width
        # box shape: nobj, seq_len, 4
        # cls shape: nobj, seq_len, 1
        # pred shape: pnobj, seq_len, 1 + 4 (prob, x0, y0, x1, y1)
        hor, box, cls = d
        seq_len, channels, height, width, nobj = *hor.shape, box.shape[0]
        imgs = hor[:, frame_idx:frame_idx+2, :, :]
        imgs = torch.stack([torch.complex(x[0], x[1]).abs() for x in imgs])

        fig = plt.figure(dpi=100)
        fig.set_size_inches((4, 3.2))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        for j, bb in enumerate(box[:, frame_idx, :]):
            x, y, x_, y_ = bb
            rect = Rectangle((x, y), x_ - x, y_ - y, linewidth=2, edgecolor='w', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y-2, 'Person%d' % j, fontsize=18, color='w')
        for k, p in enumerate(pred[:, frame_idx, :]):
            c, x, y, x_, y_ = p
            rect = Rectangle((x, y), x_ - x, y_ - y, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y-2, 'Person%d\n%.4f' % (k, c), fontsize=18, color='r')
        _ = ax.matshow(imgs[0])
        frames.append(fig2npy(fig))
        
        plt.close(fig)
        pbar.update(1)
    pbar.close()
    mpl.use(backend_) # Reset backend
    frames = np.stack(frames, axis=0)
    return frames

def generate_video(vid_name, frames, ious=None):
    nframe, h, w, channels = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(vid_name, fourcc, 10, (w, h))
    font = cv2.FONT_HERSHEY_SIMPLEX
    pbar = tqdm(total=nframe, desc='Generating %s' % vid_name)
    for i, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        
        if not ious is None:
            text = 'Box IoU: %.3f' % ious[i]
            textsize = cv2.getTextSize(text, font, 0.6, 2)[0]
            textX = (frame.shape[1] - textsize[0]) // 2
            textY = (frame.shape[0] + textsize[1]) // 6
            cv2.putText(frame, text, (textX, textY), font, 0.6, (255, 255, 255), 2)
        out.write(frame)
        pbar.update(1)
    pbar.close()
    out.release()