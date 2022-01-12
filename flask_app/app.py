from flask import Flask, render_template, request
import cv2 
import tensorflow as tf
import numpy as np

# from keras.models import load_model
# from keras.preprocessing import image

app = Flask(__name__)

labels={0:"blotch",1:"normal",2:"rotten",3:"scab"}
# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="84_percent_accuracy.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# # 
# dic = {0:"blotch",1:"normal",2:"rotten",3:"scab"}


# model = load_model('model.h5')
# model._make_predict_function()

def predict_label(img_path):
	print("hello")
	# i = image.load_img(img_path, target_size=(100,100))
	# i = image.img_to_array(i)
	# i = i.reshape(1, 100,100,3)
	# p = model.predict_classes(i)
	# return dic[p[0]]

	# path = r'static/normal1.jpeg'
	img = cv2.imread(img_path)
	new_img = cv2.resize(img, (224,224))
	print(new_img.shape)
	new_img = np.array(new_img)
	# new_img = np.transpose(new_img, (2, 0, 1))
	# input_details[0]['index'] = the index which accepts the input
	# expected type FLOAT32 for input 0, name: serving_default_input_1 

	interpreter.set_tensor(input_details[0]['index'], [new_img])

	# run the inference
	interpreter.invoke()

	# output_details[0]['index'] = the index which provides the input
	output_data = interpreter.get_tensor(output_details[0]['index'])
	minval=min(output_data[0])
	maxval=max(output_data[0])
	results=[round((i-minval)/(minval+maxval),2) for i in output_data[0]]
	output_map=[ i for i in zip(labels,results)]
	# print(round(results[2],2))
	# print("For file {}, the output is {}".format("file", results))
	return output_map


# routes
@app.route("/", methods=['GET', 'POST'])
def kuch_bhi():
	return render_template("home.html")

@app.route("/about")
def about_page():
	return "About You..!!!"






@app.route("/submit", methods = ['GET', 'POST'])
def get_hours():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)



	return render_template("home.html", prediction = p, img_path = img_path)





if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)