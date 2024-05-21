import scipy.io

# файл для исследования и тестирования .mat файлов датасета

mat1 = scipy.io.loadmat('LFPW_MAT/LFPW_image_test_0003_0.mat')
mat2 = scipy.io.loadmat('AFLW2000_MAT/image00002.mat')

#print(mat1["pt2d"]) # 3 массива по 68 точек в каждом

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

print(left_eye)
print(right_eye)

length_left_1 = (left_eye[3][0] - left_eye[0][0]) / 10
length_left_2 = (left_eye[5][1] - left_eye[1][1]) / 10
length_left_3 = (left_eye[4][1] - left_eye[2][1]) / 10
length_right_1 = (right_eye[3][0] - right_eye[0][0]) / 10
length_right_2 = (right_eye[5][1] - right_eye[1][1]) / 10
length_right_3 = (right_eye[4][1] - right_eye[2][1]) / 10

print(length_left_1, length_left_2, length_left_3, length_right_1, length_right_2, length_right_3)

print(mat2["pt2d"])