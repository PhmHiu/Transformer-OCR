import argparse
import pdb
import cv2
import os


def main(args):
	train_data = os.listdir(args.raw_train)
	valid_data = os.listdir(args.raw_valid)
	test_data = os.listdir(args.raw_test)

	max_w = 1280
	new_h = 64

	for i, image in enumerate(valid_data):
		print(i, image)
		if image == "label.txt":
			continue
		im = cv2.imread(os.path.join(args.raw_valid, image))
		h, w, d = im.shape
		unpad_im = cv2.resize(im, (int(new_h*w/h), new_h), interpolation = cv2.INTER_AREA)
		if unpad_im.shape[1] > max_w:
			print(image)
			pad_im = cv2.resize(im, (max_w, new_h), interpolation = cv2.INTER_AREA)
		else:
			pad_im = cv2.copyMakeBorder(unpad_im,0,0,0,max_w-int(new_h*w/h),cv2.BORDER_CONSTANT,value=[0,0,0])
		cv2.imwrite(os.path.join(args.unpad_valid, image), unpad_im)
		cv2.imwrite(os.path.join(args.pad_valid, image), pad_im)

	for i, image in enumerate(train_data):
		print(i, image)
		if image == "label.txt":
			continue
		im = cv2.imread(os.path.join(args.raw_train, image))
		h, w, d = im.shape
		unpad_im = cv2.resize(im, (int(new_h*w/h), new_h), interpolation = cv2.INTER_AREA)
		if unpad_im.shape[1] > max_w:
			pad_im = cv2.resize(im, (max_w, new_h), interpolation = cv2.INTER_AREA)
		else:
			pad_im = cv2.copyMakeBorder(unpad_im,0,0,0,max_w-int(new_h*w/h),cv2.BORDER_CONSTANT,value=[0,0,0])
		cv2.imwrite(os.path.join(args.unpad_train, image), unpad_im)
		cv2.imwrite(os.path.join(args.pad_train, image), pad_im)

	for i, image in enumerate(test_data):
		print(i, image)
		if image == "label.txt":
			continue
		im = cv2.imread(os.path.join(args.raw_test, image))
		h, w, d = im.shape
		unpad_im = cv2.resize(im, (int(new_h*w/h), new_h), interpolation = cv2.INTER_AREA)
		if unpad_im.shape[1] > max_w:
			print(image)
			pad_im = cv2.resize(im, (max_w, new_h), interpolation = cv2.INTER_AREA)
		else:
			pad_im = cv2.copyMakeBorder(unpad_im,0,0,0,max_w-int(new_h*w/h),cv2.BORDER_CONSTANT,value=[0,0,0])
		cv2.imwrite(os.path.join(args.unpad_test, image), unpad_im)
		cv2.imwrite(os.path.join(args.pad_test, image), pad_im)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--raw_train")
	parser.add_argument("--raw_valid")
	parser.add_argument("--raw_test")
	parser.add_argument("--pad_train")
	parser.add_argument("--pad_valid")
	parser.add_argument("--pad_test")
	parser.add_argument("--unpad_train")
	parser.add_argument("--unpad_valid")
	parser.add_argument("--unpad_test")
	args = parser.parse_args()
	main(args)

"""
	python3 resize_data.py --raw_train=/home/hieupt/Desktop/VNG/ocr/dataset/train --raw_valid=/home/hieupt/Desktop/VNG/ocr/dataset/valid --pad_train=/home/hieupt/Desktop/VNG/ocr/dataset/pad_train --pad_valid=/home/hieupt/Desktop/VNG/ocr/dataset/pad_valid --unpad_train=/home/hieupt/Desktop/VNG/ocr/dataset/unpad_train/ --unpad_valid=/home/hieupt/Desktop/VNG/ocr/dataset/unpad_valid/

"""
