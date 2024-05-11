import numpy as np
import pycocotools.mask
import cv2
import scipy.io
import os
from segment_anything_hq import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.inspection import DecisionBoundaryDisplay

def show_res(masks, scores, image):
    data = []
    target = 0
    for i, (mask, score) in enumerate(zip(masks, scores)):
        data, target = mask_estimation(mask, image, score)
    return data, target

def get_keypoints_coords(path):
    mat1 = scipy.io.loadmat(path)

    left_eye = [[mat1['pt3d_68'][0][36], mat1['pt3d_68'][1][36]],
                [mat1['pt3d_68'][0][37], mat1['pt3d_68'][1][37]],
                [mat1['pt3d_68'][0][38], mat1['pt3d_68'][1][38]],
                [mat1['pt3d_68'][0][39], mat1['pt3d_68'][1][39]],
                [mat1['pt3d_68'][0][40], mat1['pt3d_68'][1][40]],
                [mat1['pt3d_68'][0][41], mat1['pt3d_68'][1][41]]]

    right_eye = [[mat1['pt3d_68'][0][42], mat1['pt3d_68'][1][42]],
                 [mat1['pt3d_68'][0][43], mat1['pt3d_68'][1][43]],
                 [mat1['pt3d_68'][0][44], mat1['pt3d_68'][1][44]],
                 [mat1['pt3d_68'][0][45], mat1['pt3d_68'][1][45]],
                 [mat1['pt3d_68'][0][46], mat1['pt3d_68'][1][46]],
                 [mat1['pt3d_68'][0][47], mat1['pt3d_68'][1][47]]]

    return left_eye, right_eye


def get_eye_coords(left_eye, right_eye):
    length_left_1 = (left_eye[3][0] - left_eye[0][0]) / 9  # расстояние между крайней левой и крайней правой точкой левого глаза
    length_left_2 = (left_eye[5][1] - left_eye[1][1]) / 3  # расстояние между верхней и нижней точками левого глаза
    length_left_3 = (left_eye[4][1] - left_eye[2][1]) / 3  # расстояние между верхней и нижней точками левого глаза
    length_right_1 = (right_eye[3][0] - right_eye[0][0]) / 9  # расстояние между крайней левой и крайней правой точкой правого глаза
    length_right_2 = (right_eye[5][1] - right_eye[1][1]) / 3  # расстояние между верхней и нижней точками правого глаза
    length_right_3 = (right_eye[4][1] - right_eye[2][1]) / 3  # расстояние между верхней и нижней точками правого глаза
    middle_left_1 = left_eye[1][1] + (left_eye[5][1] - left_eye[1][1]) / 2  # координата середины глаза для левой крайней точки левого глаза
    middle_left_2 = left_eye[2][1] + (left_eye[4][1] - left_eye[2][1]) / 2  # координата середины глаза для правой крайней точки левого глаза
    middle_right_1 = right_eye[1][1] + (right_eye[5][1] - right_eye[1][1]) / 2  # координата середины глаза для левой крайней точки правого глаза
    middle_right_2 = right_eye[2][1] + (right_eye[4][1] - right_eye[2][1]) / 2  # координата середины глаза для правой крайней точки правого глаза

    eye_coords = [[left_eye[0][0] + length_left_1, middle_left_1],
                  [left_eye[1][0], left_eye[1][1] + length_left_2],
                  [left_eye[2][0], left_eye[2][1] + length_left_3],
                  [left_eye[3][0] - length_left_1, middle_left_2],
                  [left_eye[4][0], left_eye[4][1] - length_left_2],
                  [left_eye[5][0], left_eye[5][1] - length_left_3],
                  [right_eye[0][0] + length_right_1, middle_right_1],
                  [right_eye[1][0], right_eye[1][1] + length_right_2],
                  [right_eye[2][0], right_eye[2][1] + length_right_3],
                  [right_eye[3][0] - length_right_1, middle_right_2],
                  [right_eye[4][0], right_eye[4][1] - length_right_2],
                  [right_eye[5][0], right_eye[5][1] - length_right_3]]

    return eye_coords


def mask_estimation(mask, image, score):
    encoded_mask = pycocotools.mask.encode(np.asfortranarray(mask))
    mask_area = pycocotools.mask.area(encoded_mask)
    mask_bbox = pycocotools.mask.toBbox(encoded_mask)  # bbox = [x, y, w, h]
    height, width, channels = image.shape  # картинка должна открываться через cv2
    image_area = width * height
    mask_percentage = (mask_area / image_area) * 100  # процент заполнения картинки маской
    bbox_width = mask_bbox[2]
    bbox_height = mask_bbox[3]

    if bbox_width >= bbox_height:
        bbox_area = bbox_height / bbox_width
        bbox_percentage = bbox_area * 100  # считаем процент соотношения сторон bbox
    else:
        bbox_area = bbox_width / bbox_height
        bbox_percentage = bbox_area * 100  # считаем процент соотношения сторон bbox

    mask_bbox_percentage = bbox_area / mask_area * 100

    if mask_percentage < 0.1 and bbox_percentage >= 85:
        estimation = 0  # best quality(should be iris mask)
        return [mask_percentage, bbox_percentage], estimation
    elif mask_percentage < 0.15 and bbox_percentage >= 65:
        estimation = 1  # good quality(iris mask probably)
        return [mask_percentage, bbox_percentage], estimation
    elif mask_percentage < 0.5 and bbox_percentage >= 40:
        estimation = 2  # moderate(eye mask probably)
        return [mask_percentage, bbox_percentage], estimation
    else:
        estimation = 3  # bad mask
        return [mask_percentage, bbox_percentage], estimation


if __name__ == "__main__":
    sam_checkpoint = "sam_hq_vit_h.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    hq_token_only = True

    mat_folder_path = "AFLW2000_MAT"
    pics_folder_path = "AFLW2000_PICS"

    directory = os.fsencode("AFLW2000_PICS")
    dataset_data = []
    dataset_target = []

    for file in os.listdir(directory):
        file_name = os.fsdecode(file).split(".")[0]

        image = cv2.imread('AFLW2000_PICS/' + file_name + ".jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.set_image(image)

        left_eye, right_eye = get_keypoints_coords("AFLW2000_MAT/" + file_name + ".mat")
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

            data, target = show_res(masks, scores, image)
            dataset_data.append(data)
            dataset_target.append(target)

    print(dataset_data)
    print(len(dataset_data))
    print(dataset_target)
    print(len(dataset_target))

    numpy_data = np.array(dataset_data)
    numpy_target = np.array(dataset_target)

    C = 1.0  # SVM regularization parameter
    models = (
        svm.SVC(kernel="linear", C=C),
        svm.LinearSVC(C=C, max_iter=10000, dual="auto"),
        svm.SVC(kernel="rbf", gamma=0.7, C=C),
        svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),
    )
    models = (clf.fit(numpy_data, numpy_target) for clf in models)

    titles = (
        "SVC with linear kernel",
        "LinearSVC (linear kernel)",
        "SVC with RBF kernel",
        "SVC with polynomial (degree 3) kernel",
    )

    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = numpy_data[:, 0], numpy_data[:, 1]

    for clf, title, ax in zip(models, titles, sub.flatten()):
        disp = DecisionBoundaryDisplay.from_estimator(
            clf,
            numpy_data,
            response_method="predict",
            cmap=plt.cm.coolwarm,
            alpha=0.8,
            ax=ax,
            xlabel="mask_percentage",
            ylabel="bbox_percentage",
        )
        ax.scatter(X0, X1, c=numpy_target, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.show()