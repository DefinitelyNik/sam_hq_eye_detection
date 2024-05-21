import numpy as np
import pycocotools.mask
import torch
import matplotlib.pyplot as plt
import cv2
import scipy.io
from segment_anything_hq import SamPredictor, sam_model_registry
from PIL import Image
from pycocotools.coco import COCO


def show_mask(mask, ax, random_color=False, save_mask=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    if save_mask:
        #im = Image.fromarray(mask_image)
        #im.save("mask.png")
        # with open("mask1.txt", "w") as output:
        #     output.write(str())
        encoded_mask = pycocotools.mask.encode(np.asfortranarray(mask))
        print(pycocotools.mask.area(encoded_mask))
        print("")
        print(pycocotools.mask.toBbox(encoded_mask))

def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


def show_res(masks, scores, input_point, input_label, input_box, image, file_name, index):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), save_mask=True)
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None):
            show_points(input_point, input_label, plt.gca())
        file_name1 = file_name.split('.')[0]
        plt.savefig("pics/" + file_name1 + "_" + str(index) + ".png")
        print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.show()


def show_res_multi(masks, scores, input_point, input_label, input_box, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_box:
        show_box(box, plt.gca())
    for score in scores:
        print(f"Score: {score:.3f}")
    plt.axis('off')
    plt.show()

def get_keypoints_coords(path):
    mat1 = scipy.io.loadmat(path)

    left_eye = [[mat1['pt2d'][0][36], mat1['pt2d'][1][36]],
                [mat1['pt2d'][0][37], mat1['pt2d'][1][37]],
                [mat1['pt2d'][0][38], mat1['pt2d'][1][38]],
                [mat1['pt2d'][0][39], mat1['pt2d'][1][39]],
                [mat1['pt2d'][0][40], mat1['pt2d'][1][40]],
                [mat1['pt2d'][0][41], mat1['pt2d'][1][41]]]

    right_eye = [[mat1['pt2d'][0][42], mat1['pt2d'][1][42]],
                 [mat1['pt2d'][0][43], mat1['pt2d'][1][43]],
                 [mat1['pt2d'][0][44], mat1['pt2d'][1][44]],
                 [mat1['pt2d'][0][45], mat1['pt2d'][1][45]],
                 [mat1['pt2d'][0][46], mat1['pt2d'][1][46]],
                 [mat1['pt2d'][0][47], mat1['pt2d'][1][47]]]

    return left_eye, right_eye

def get_eye_coords(left_eye, right_eye):
    length_left_1 = (left_eye[3][0] - left_eye[0][0]) / 10
    length_left_2 = (left_eye[5][1] - left_eye[1][1]) / 10
    length_left_3 = (left_eye[4][1] - left_eye[2][1]) / 10
    length_right_1 = (right_eye[3][0] - right_eye[0][0]) / 10
    length_right_2 = (right_eye[5][1] - right_eye[1][1]) / 10
    length_right_3 = (right_eye[4][1] - right_eye[2][1]) / 10

    eye_coords = [[left_eye[0][0] + length_left_1, left_eye[0][1]],
                  [left_eye[1][0], left_eye[1][1] + length_left_2],
                  [left_eye[2][0], left_eye[2][1] + length_left_3],
                  [left_eye[3][0] - length_left_1, left_eye[3][1]],
                  [left_eye[4][0], left_eye[4][1] - length_left_2],
                  [left_eye[5][0], left_eye[5][1] - length_left_3],
                  [right_eye[0][0] + length_right_1, right_eye[0][1]],
                  [right_eye[1][0], right_eye[1][1] + length_right_2],
                  [right_eye[2][0], right_eye[2][1] + length_right_3],
                  [right_eye[3][0] - length_right_1, right_eye[3][1]],
                  [right_eye[4][0], right_eye[4][1] - length_right_2],
                  [right_eye[5][0], right_eye[5][1] - length_right_3]]

    return eye_coords

if __name__ == "__main__":
    sam_checkpoint = "sam_hq_vit_h.pth"
    #sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    hq_token_only = True

    mat_folder_path = "LFPW_MAT"
    pics_folder_path = "LFPW_PICS"

    file_name = 'LFPW_image_test_0050_0'

    image = cv2.imread('LFPW_PICS/' + file_name + ".jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)

    left_eye, right_eye = get_keypoints_coords("LFPW_MAT/" + file_name + ".mat")
    eye_coords = get_eye_coords(left_eye, right_eye)
    file_name += ".jpg"

    for i in range(len(eye_coords)):
        input_point = np.array([eye_coords[i]])
        input_label = np.ones(input_point.shape[0])
        input_box = None

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False,
            hq_token_only=hq_token_only,
        )

        show_res(masks,scores,input_point, input_label, input_box, image, file_name, i)